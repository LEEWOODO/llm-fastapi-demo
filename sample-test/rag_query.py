from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

load_dotenv()

# 1. 사용자 질문 입력
query = input("질문을 입력하세요:")

# 2. 임베딩 모델 로딩
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. 벡터스토어 연결
db = OpenSearchVectorSearch(
    index_name="rag-index2",
    embedding_function=embedding,
    opensearch_url="http://localhost:9200",
    http_auth=("admin", "admin")
)

print("🟡 임베딩 준비 완료!")

# 4. 유사 문서 검색
docs = db.similarity_search(query, k=3)
context = "\n".join([doc.page_content for doc in docs])
print("🔍 검색된 문서 수:", len(docs))
print("🔍 검색된 문서 내용:", context)


# 5. 프롬프트 구성
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="{context} 이 내용을 바탕으로 '{question}'에 대해 설명해줘."
)

# 6. LLM 설정 (지원되는 모델 사용)
qa_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",  # ✅ CPU에서 안정적으로 작동
    device=-1
)

llm = HuggingFacePipeline(
    pipeline=qa_pipeline,
    model_kwargs={"max_new_tokens": 100, "temperature": 0.7}
)

# 7. 파이프라인 구성 및 실행
chain = RunnableLambda(lambda question: FalconLLMProvider.invoke(
    prompt_template.format(context=context, question=question)
))

response = chain.invoke(query)
print("\n🤖 답변:", response.content if hasattr(response, "content") else response)
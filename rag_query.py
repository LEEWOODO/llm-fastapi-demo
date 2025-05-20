from dotenv import load_dotenv
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

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
    template="""
    다음 문서를 참고하여 질문에 답해주세요:
    ---
    {context}
    ---
    질문: {question}
    답변:
    """
)

# 6. LLM 설정 (Groq 또는 OpenAI 중 선택)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0
)

# 7. 파이프라인 구성 및 실행
chain = RunnableLambda(lambda question: llm.invoke(
    prompt_template.format(context=context, question=question)
))

response = chain.invoke(query)
print("\n🤖 답변:", response.content)
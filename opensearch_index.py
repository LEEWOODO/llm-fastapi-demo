from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# 1. 문서 로드
loader = TextLoader("docs/sample.txt")
documents = loader.load()

# 2. 문장 분할
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# 3. 임베딩 모델 (무료)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. OpenSearch에 저장
vectorstore = OpenSearchVectorSearch.from_documents(
    chunks,
    embedding=embedding,  # ✅ 올바르게 수정
    opensearch_url="http://localhost:9200",
    index_name="rag-index"
)

print(f"✅ {len(chunks)}개 문서를 OpenSearch에 인덱싱 완료!")

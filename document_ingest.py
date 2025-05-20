from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
import os

# 1. 문서 로딩
loader = TextLoader("docs/sample.txt")
documents = loader.load()

# 2. 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# 3. 임베딩 모델 불러오기 (무료 모델 사용)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. OpenSearch 연결
opensearch_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False,
    verify_certs=False
)

# 5. 벡터스토어 생성 또는 연결
db = OpenSearchVectorSearch(
    index_name="rag-index2",
    embedding_function=embedding,
    opensearch_url="http://localhost:9200",
    http_auth=("admin", "admin")
)

# 6. 문서 임베딩 후 저장
db.add_documents(split_docs)
print("✅ 문서가 벡터화되어 OpenSearch에 저장되었습니다.")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import ElasticsearchStore
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

import os

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

# 1. 문서 로딩
loader = TextLoader("../../docs/sample.txt")
documents = loader.load()

# 2. 문장 분할
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 3. 임베딩 생성 (OpenAI 사용)
# embedding = OpenAIEmbeddings()
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. ElasticsearchStore에 저장
vectorstore = ElasticsearchStore.from_documents(
    docs,
    embedding,
    es_url="http://localhost:9200",
    index_name="rag-index"
)

print(f"✅ 문서 {len(docs)}개 Elasticsearch에 저장 완료")

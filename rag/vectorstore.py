from langchain_community.vectorstores import OpenSearchVectorSearch

from rag.embeddings import embedding

vectorstore = OpenSearchVectorSearch(
    index_name="rag-index",
    embedding_function=embedding,
    opensearch_url="http://localhost:9200"
)

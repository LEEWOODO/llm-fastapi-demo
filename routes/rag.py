from fastapi import APIRouter
from rag.chains import rag_chain
from models.schema import RAGQuery

router = APIRouter()

@router.post("/rag")
def ask_rag(query: RAGQuery):
    result = rag_chain.invoke({"query": query.query})
    return {
        "query": query.query,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }

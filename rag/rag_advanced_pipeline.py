from fastapi import APIRouter

from langgraph.langgraph_workflow_advanced import rag_graph
from models.schema import AdvRAGQuery

router = APIRouter()


@router.post("/adv_rag")
def run_advanced_rag(q: AdvRAGQuery):
    result = rag_graph.invoke({"query": q.query})
    return {
        "query": q.query,
        "answer": result["answer"],
        "docs": result.get("reranked", [])
    }

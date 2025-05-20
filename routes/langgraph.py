from fastapi import APIRouter

from langgraph_flow import langgraph_graph
from models.schema import Question

router = APIRouter()


@router.post("/langgraph")
def run_langgraph(q: Question):
    return langgraph_graph.invoke({"question": q.prompt})

from fastapi import APIRouter

from agent.agent_executor import agent_executor
from models.schema import AgentQuery

router = APIRouter()


@router.post("/agent")
def ask_agent(query: AgentQuery):
    result = agent_executor.invoke({"input": query.input})
    return {"result": result["output"]}

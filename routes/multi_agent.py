from fastapi import APIRouter
from models.schema import AgentQuery
from langgraph.multi_agent_flow import multi_agent_app

router = APIRouter()

@router.post("/multi-agent")
def ask_multi_agent(query: AgentQuery):
    """
    ✅ 다중 Agent 흐름 API
    - 검색 → 계산 → 요약 단계를 순차적으로 처리
    - 최종 요약 결과 반환
    """
    result = multi_agent_app.invoke({"query": query.input})
    return {"result": result["final_summary"]}

from typing import TypedDict
from rag.rag import rag_chain

class MultiAgentState(TypedDict):
    query: str
    search_result: str


def search_agent_node(state: MultiAgentState) -> MultiAgentState:
    """
    🔍 RAG 기반 검색 Agent 노드
    - 사용자 쿼리를 기반으로 관련 문서를 검색하고 응답 생성
    - 결과를 state['search_result']에 저장
    """
    query = state["query"]
    result = rag_chain.invoke({"query": query})
    state["search_result"] = result["result"]
    return state

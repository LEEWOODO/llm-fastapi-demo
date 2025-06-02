from typing import TypedDict
from rag.rag import rag_chain

class MultiAgentState(TypedDict):
    query: str
    search_result: str


def truncate_text(text: str, max_tokens: int = 1800) -> str:
    # 너무 길 경우 앞쪽 일부만 retain (tokenizer 안 쓰고 단순 토큰 수 기반 처리)
    return ' '.join(text.split()[:max_tokens])


def search_agent_node(state: MultiAgentState) -> MultiAgentState:
    """
    🔍 RAG 기반 검색 Agent 노드
    - 사용자 쿼리를 기반으로 관련 문서를 검색하고 응답 생성
    - 결과를 state['search_result']에 저장
    """
    query = state["query"]
    result = rag_chain.invoke({"query": query})

    answer = result["result"]
    answer = truncate_text(answer, max_tokens=1800)  # ✅ LLM input에 안전한 길이

    state["search_result"] = answer
    return state

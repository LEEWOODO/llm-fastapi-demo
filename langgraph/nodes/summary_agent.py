from typing import TypedDict, Optional
from rag.llmchain import summarizer_chain

class MultiAgentState(TypedDict):
    query: str
    search_result: Optional[str]
    calc_result: Optional[str]
    final_summary: Optional[str]


def summary_agent_node(state: MultiAgentState) -> MultiAgentState:
    """
    📝 요약 Agent 노드
    - 앞서 수집된 검색/계산 결과를 종합하여 요약
    - 결과를 state['final_summary']에 저장
    """
    content = f"검색 결과: {state.get('search_result', '')}\n계산 결과: {state.get('calc_result', '')}"

    result = summarizer_chain.invoke({"input": content})

    # state["final_summary"] = result
    state["final_summary"] = result["text"]
    return state

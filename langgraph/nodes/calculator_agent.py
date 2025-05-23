from typing import TypedDict, Optional
from agent.agent_executor import agent_executor

class MultiAgentState(TypedDict):
    query: str
    search_result: Optional[str]
    calc_result: Optional[str]


def calculator_agent_node(state: MultiAgentState) -> MultiAgentState:
    """
    🧮 계산 Agent 노드
    - 쿼리에 "곱하기", "더하기" 등의 키워드가 포함된 경우 계산 실행
    - 결과를 state['calc_result']에 저장
    """
    query = state["query"]
    if any(op in query for op in ["곱하기", "더하기", "나누기", "빼기"]):
        result = agent_executor.invoke({"input": query})
        state["calc_result"] = result["output"]
    return state

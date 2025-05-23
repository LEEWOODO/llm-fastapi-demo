from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from langgraph.nodes.search_agent import search_agent_node
from langgraph.nodes.calculator_agent import calculator_agent_node
from langgraph.nodes.summary_agent import summary_agent_node

# -------------------------
# ✅ 상태 정의
# -------------------------
class MultiAgentState(TypedDict):
    query: str
    search_result: Optional[str]
    calc_result: Optional[str]
    final_summary: Optional[str]

# -------------------------
# ✅ LangGraph 흐름 정의
# -------------------------
graph = StateGraph(MultiAgentState)

# 노드 추가
graph.add_node("search", search_agent_node)
graph.add_node("calculate", calculator_agent_node)
graph.add_node("summarize", summary_agent_node)

# 흐름 구성
graph.set_entry_point("search")
graph.add_edge("search", "calculate")
graph.add_edge("calculate", "summarize")
graph.add_edge("summarize", END)

# 실행기 컴파일
multi_agent_app = graph.compile()

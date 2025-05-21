import re
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END


# ------------------------------
# ✅ 상태 정의
# ------------------------------
class QAState(TypedDict):
    question: str
    answer: str


# ------------------------------
# ✅ 질문 분류 노드 (분기 기준)
# ------------------------------
def classify_question(state: QAState) -> Literal["math", "rag"]:
    q = state["question"].strip().lower()
    print(f"💡 classify_question: '{q}'")
    if re.search(r"\d+\s*[\*×x곱하기]\s*\d+", q):
        print("🔀 classified as: math")
        return "math"
    print("🔀 classified as: rag")
    return "rag"


# ------------------------------
# ✅ Agent 처리 노드
# ------------------------------
from agent.agent_executor import agent_executor


def agent_node(state: QAState) -> QAState:
    question = state["question"]
    result = agent_executor.invoke({"input": question})
    return {"question": question, "answer": result["output"]}


# ------------------------------
# ✅ RAG 처리 노드 (mock)
# ------------------------------
from rag.chains import rag_chain


def rag_node(state: QAState) -> QAState:
    question = state["question"]
    result = rag_chain.invoke({"query": question})
    return {"question": question, "answer": result["result"]}


# ------------------------------
# ✅ 그래프 구성
# ------------------------------
builder = StateGraph(QAState)

builder.add_node("agent", agent_node)
builder.add_node("rag", rag_node)

builder.set_conditional_entry_point(
    classify_question,
    {"math": "agent", "rag": "rag"}
)

builder.add_edge("agent", END)
builder.add_edge("rag", END)

langgraph_graph = builder.compile()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
import re

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
def agent_node(state: QAState) -> QAState:
    question = state["question"]
    try:
        result = eval(question.replace("x", "*").replace("\u00D7", "*"))  # 간단 계산만 허용
        return {"question": question, "answer": str(result)}
    except Exception:
        return {"question": question, "answer": "계산 오류"}

# ------------------------------
# ✅ RAG 처리 노드 (mock)
# ------------------------------
def rag_node(state: QAState) -> QAState:
    # TODO: 실제 rag_chain.invoke 연결 예정
    return {"question": state["question"], "answer": "[RAG] 여기에 문서 기반 답변이 생성됩니다."}

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

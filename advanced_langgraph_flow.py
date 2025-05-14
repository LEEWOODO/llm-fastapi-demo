from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, List

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from main import llm  # 기존에 정의된 HuggingFacePipeline 또는 Groq 기반 LLM

# ✅ 검색 전용으로 vectorstore 생성
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = OpenSearchVectorSearch(
    index_name="rag-index",
    embedding_function=embedding,
    opensearch_url="http://localhost:9200"
)

# ------------------------------
# ✅ 상태 정의
# ------------------------------
class RAGState(TypedDict):
    query: str
    docs: List[tuple]  # 또는 List[Tuple[Document, float]]
    reranked: List[dict]
    answer: str

# 🔍 (1) Retrieval Node
def retrieve_node(state: RAGState) -> RAGState:
    docs_with_scores = vectorstore.similarity_search_with_score(state["query"], k=10)

    # ✅ score 포함된 튜플로 저장
    return {**state, "docs": docs_with_scores}

# 🧠 (2) Rerank Node (간단 필터 또는 LLM rerank)
RERANK_PROMPT = """다음은 사용자 질문과 관련된 문서입니다.
각 문서가 질문에 얼마나 관련이 있는지 [0.0 ~ 1.0] 사이의 점수로 평가하세요.

질문: "{query}"

문서:
{doc}

관련도 점수 (숫자만):"""
def rerank_node(state: RAGState) -> RAGState:
    scored = []

    for doc, score in state["docs"]:  # 기존 유사도 score도 함께 있음
        prompt = RERANK_PROMPT.format(query=state["query"], doc=doc.page_content)

        try:
            response = llm.invoke(prompt)
            # LLM 응답에서 숫자만 추출 (예: "0.9")
            rerank_score = float(response.strip())
        except Exception as e:
            print(f"⚠️ Rerank 오류: {e}")
            rerank_score = 0.0

        scored.append({
            "content": doc.page_content,
            "original_score": score,
            "rerank_score": rerank_score
        })

    # 🔼 rerank_score 기준 내림차순 정렬
    reranked = sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:2]
    return {**state, "reranked": reranked}

# ✍️ (3) Answer Node
def answer_node(state: RAGState) -> RAGState:
    docs_text = "\n\n".join([
        f"[rerank_score={doc['rerank_score']:.2f}] {doc['content']}"
        for doc in state["reranked"]
    ])
    prompt = (
        "아래 문서를 참고하여 질문에 답변해주세요.\n\n"
        f"{docs_text}\n\n"
        f"질문: {state['query']}\n"
        "답변:"
    )
    result = llm.invoke(prompt)
    return {**state, "answer": result}

# ------------------------------
# ✅ LangGraph 구성
# ------------------------------
builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("rerank", rerank_node)
builder.add_node("generate_answer", answer_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate_answer")
builder.add_edge("generate_answer", END)

rag_graph = builder.compile()

# 테스트 실행
if __name__ == "__main__":
    result = rag_graph.invoke({"query": "LLM 에 대해 알려죠"})
    print(f"💡 answer: '{result['answer']}'")

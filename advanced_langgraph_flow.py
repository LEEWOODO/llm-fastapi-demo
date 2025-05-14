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
    docs: List[str]
    reranked: List[str]
    answer: str

# 🔍 (1) Retrieval Node
def retrieve_node(state: RAGState) -> RAGState:
    docs = vectorstore.similarity_search(state["query"], k=5)
    texts = [doc.page_content for doc in docs]
    return {**state, "docs": texts}

# 🧠 (2) Rerank Node (간단 필터 또는 LLM rerank)
def rerank_node(state: RAGState) -> RAGState:
    # 임시 간단 로직: 길이 순서 정렬
    reranked = sorted(state["docs"], key=len, reverse=True)[:2]
    return {**state, "reranked": reranked}

# ✍️ (3) Answer Node
def answer_node(state: RAGState) -> RAGState:
    prompt = (
        "아래 문서를 참고하여 질문에 답변해주세요.\n\n"
        f"문서:\n{chr(10).join(state['reranked'])}\n\n"
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

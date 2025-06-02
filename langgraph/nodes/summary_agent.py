import re
from typing import TypedDict, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm.summarizer_provider import SummarizerProvider


class MultiAgentState(TypedDict):
    query: str
    search_result: Optional[str]
    calc_result: Optional[str]
    final_summary: Optional[str]

# ✅ 사전 훈련된 다국어 임베딩 모델
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def extract_similar_sentences(text: str, query: str, threshold: float = 0.3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    embeddings = embedder.encode(sentences + [query])
    sentence_vectors, query_vector = embeddings[:-1], embeddings[-1:]

    scores = cosine_similarity(query_vector, sentence_vectors)[0]
    selected = [s for s, score in zip(sentences, scores) if score >= threshold]

    return "\n".join(selected)

def is_math_query(query: str) -> bool:
    return any(op in query for op in ["곱하기", "더하기", "나누기", "빼기", "*", "/", "+", "-"])

def clean_search_result(text: str, query: str) -> str:
    # ✅ 시스템 프롬프트 제거 (전체 블록까지)
    text = re.sub(
        r"Use the following pieces of context to answer the question at the end\..*?Helpful Answer:",
        "",
        text,
        flags=re.DOTALL,
    )
    text = text.replace("Question:", "").replace("Answer:", "").strip()

    # 🔁 유사 문장 추출
    filtered = extract_similar_sentences(text, query)
    return filtered if filtered.strip() else text.strip()


def truncate_text(text: str, max_tokens: int = 1800) -> str:
    # 너무 길 경우 앞쪽 일부만 retain (tokenizer 안 쓰고 단순 토큰 수 기반 처리)
    return ' '.join(text.split()[:max_tokens])


def summary_agent_node(state: MultiAgentState) -> MultiAgentState:
    """
    📝 요약 Agent 노드
    - 앞서 수집된 검색/계산 결과를 종합하여 요약
    - 결과를 state['final_summary']에 저장
    """
    query = state["query"]
    search = state.get("search_result", "")
    calc = state.get("calc_result", "")

    if is_math_query(query):
        state["final_summary"] = f"계산 결과는 {calc}입니다."
        return state

    cleaned_search = clean_search_result(search, query)
    # cleaned_search = truncate_text(cleaned_search, 1800)
    
    # ✅ 너무 짧으면 요약하지 말고 그대로 사용
    if len(cleaned_search.split()) <= 5:
        state["final_summary"] = cleaned_search
        return state

    # 요약 실행
    content = f"다음 정보를 1~2문장으로 간결하게 요약해줘:\n\n{cleaned_search}"
    result = SummarizerProvider().get_chain().invoke({"input": content})
    state["final_summary"] = result["text"]
    return state

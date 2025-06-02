from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

from llm.provider import FalconLLMProvider
from rag.vectorstore import vectorstore

# 1. 개별 컴포넌트 정의
prompt = PromptTemplate.from_template(
    "다음 문서를 참고하여 질문에 답해줘:\n\n{context}\n\n질문: {question}"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = FalconLLMProvider().get_chain()


# 2. 유틸리티 함수들
def extract_query(input_data):
    """입력에서 쿼리 추출"""
    if isinstance(input_data, dict):
        return input_data.get("query", str(input_data))
    return str(input_data)


def format_docs(docs):
    """문서 리스트를 하나의 텍스트로 결합"""
    return "\n\n".join(doc.page_content for doc in docs)


def add_sources(result_and_docs):
    """LLM 결과에 소스 문서 추가"""
    return {
        "result": result_and_docs["answer"],
        "source_documents": result_and_docs["docs"]
    }


# 3. ✅ LCEL 스타일 RAG 체인
rag_chain = (
        # Step 1: 쿼리 추출
        RunnableLambda(extract_query)

        # Step 2: 병렬로 쿼리와 관련 문서 가져오기
        | RunnableParallel({
            "query": RunnablePassthrough(),  # 쿼리를 그대로 전달
            "docs": retriever  # 관련 문서 검색
        })

        # Step 3: 프롬프트를 위한 데이터 준비
        | RunnableParallel({
            "context": RunnableLambda(lambda x: format_docs(x["docs"])),  # 문서를 텍스트로 변환
            "question": RunnableLambda(lambda x: x["query"]),  # 쿼리 추출
            "docs": RunnableLambda(lambda x: x["docs"])  # 원본 문서 보존
        })

        # Step 4: 프롬프트 생성과 LLM 호출을 병렬로
        | RunnableParallel({
            "answer": prompt | llm,  # 프롬프트 → LLM
            "docs": RunnableLambda(lambda x: x["docs"])  # 문서 정보 보존
        })

        # Step 5: 최종 결과 포맷팅
        | RunnableLambda(add_sources)
)

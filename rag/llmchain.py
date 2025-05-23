from fastapi import APIRouter
from langchain.prompts import PromptTemplate

from llm.provider import FalconLLMProvider
from models.schema import LLMChainQuery
from langchain_core.runnables import RunnableLambda
from llm.summarizer_provider import SummarizerProvider
router = APIRouter()

# ✅ 요약용 프롬프트
# summary_prompt = PromptTemplate.from_template(
#     "다음 내용을 간결하게 요약해줘:\n\n{input}"
# )
summary_prompt = PromptTemplate.from_template(
    "입력된 정보를 간단하게 3줄 이하로 요약해줘:\n\n{input}"
)


# ✅ 요약 체인 정의
# summarizer_chain = summary_prompt | llm
# HuggingFace LLM이 string 리턴하므로 래핑해서 딕셔너리로 변환
# summarizer_chain = summary_prompt | llm | RunnableLambda(lambda x: {"text": x})
summarizer_chain = SummarizerProvider().get_chain()

@router.post("/llmchain")
def ask_llmchain(query: LLMChainQuery):
    result = summarizer_chain.invoke({"name": query.name})
    return {"message": result}

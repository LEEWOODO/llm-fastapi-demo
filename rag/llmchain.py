from fastapi import APIRouter
from langchain.prompts import PromptTemplate

from llm.provider import llm
from models.schema import LLMChainQuery

router = APIRouter()

# 프롬프트 템플릿
prompt = PromptTemplate.from_template("안녕하세요, {name}님. 무엇을 도와드릴까요?")
llm_chain = prompt | llm


@router.post("/llmchain")
def ask_llmchain(query: LLMChainQuery):
    result = llm_chain.invoke({"name": query.name})
    return {"message": result}

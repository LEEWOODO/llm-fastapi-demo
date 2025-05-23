from fastapi import APIRouter

from llm.summarizer_provider import SummarizerProvider
from models.schema import LLMChainQuery

router = APIRouter()

@router.post("/llmchain")
def ask_llmchain(query: LLMChainQuery):
    result = SummarizerProvider().get_chain().invoke({"input": query.name})
    return {"message": result}

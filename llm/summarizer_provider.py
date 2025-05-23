from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from .base import LLMProvider

class SummarizerProvider(LLMProvider):
    def __init__(self):
        pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        llm = HuggingFacePipeline(pipeline=pipe)
        prompt = PromptTemplate.from_template(
            "다음 정보를 요약해줘. 중요한 내용만 추려서 3줄 이내로 정리해줘:\n\n{input}"
        )
        self.chain = prompt | llm | RunnableLambda(lambda x: {"text": x})

    def get_chain(self):
        return self.chain

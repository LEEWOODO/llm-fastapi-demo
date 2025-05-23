from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from .base import LLMProvider

class SummarizerProvider(LLMProvider):
    def __init__(self):
        # pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        pipe = pipeline("summarization", model="t5-small")

        llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"max_new_tokens": 100, "temperature": 0.7, "max_length": 100}
        )
        prompt = PromptTemplate.from_template(
            "다음 정보를 읽고, 핵심만 간결하게 요약해줘. 중복된 문장, 시스템 메시지는 제거하고, 요점을 2~3문장으로 정리해줘:\n\n{input}"
        )
        self.chain = prompt | llm | RunnableLambda(lambda x: {"text": x})

    def get_chain(self):
        return self.chain

import os
from typing import ClassVar

from groq import Groq
from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# ✅ HuggingFace LLM
qa_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    device=-1
)

llm = HuggingFacePipeline(
    pipeline=qa_pipeline,
    model_kwargs={"max_new_tokens": 100, "temperature": 0.7}
)


# ✅ Groq 기반 LLM
class GroqAgentLLM(LLM):
    model_name: ClassVar[str] = "llama3-8b-8192"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

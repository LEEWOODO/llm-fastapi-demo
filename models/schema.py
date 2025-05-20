# 🎯 공통 스키마는 models/schema.py로 이동
from pydantic import BaseModel

class RAGQuery(BaseModel):
    query: str

class Question(BaseModel):
    prompt: str

class AgentQuery(BaseModel):
    input: str

class LLMChainQuery(BaseModel):
    input: str

class AdvRAGQuery(BaseModel):
    input: str


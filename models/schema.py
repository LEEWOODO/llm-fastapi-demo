# 🎯 공통 스키마는 models/schema.py로 이동
from pydantic import BaseModel


class Item(BaseModel):
    id: int
    name: str
    description: str


class RAGQuery(BaseModel):
    query: str


class Question(BaseModel):
    prompt: str


class AgentQuery(BaseModel):
    input: str


class LLMChainQuery(BaseModel):
    name: str


class AdvRAGQuery(BaseModel):
    query: str

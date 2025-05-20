import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abc import ABC, abstractmethod
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from openai import OpenAI, OpenAIError
from pydantic import BaseModel
from agent.agent_executor import agent_executor  # ✅ 정상 작동해야 함


from fastapi import APIRouter
# from rag.chains import rag_chain
from models.schema import RAGQuery, AdvRAGQuery, LLMChainQuery, Question, AgentQuery

load_dotenv()
app = FastAPI()

# ------------------------------
# ✅ 데이터 모델 정의
# ------------------------------

class Item(BaseModel):
    id: int
    name: str
    description: str

items_db: List[Item] = []

# ------------------------------
# ✅ RESTful API
# ------------------------------

@app.get("/items")
def get_items():
    return items_db

@app.get("/items/{item_id}")
def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/items")
def create_item(item: Item):
    items_db.append(item)
    return item

@app.put("/items/{item_id}")
def update_item(item_id: int, updated_item: Item):
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            items_db[idx] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            items_db.remove(item)
            return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")

# ------------------------------
# ✅ 전략 패턴: LLMProvider 인터페이스
# ------------------------------

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            raise RuntimeError("OpenAI 사용 불가") from e

class GroqProvider(LLMProvider):
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def chat(self, prompt: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                model="llama3-8b-8192",  # 또는 사용 가능한 다른 Groq 모델
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"[Groq 오류] {str(e)}"

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ------------------------------
# ✅ OpenSearch 기반 벡터 검색 세팅
# ------------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = OpenSearchVectorSearch(
    index_name="rag-index",
    embedding_function=embedding,
    opensearch_url="http://localhost:9200"
)

qa_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",  # ✅ CPU에서 안정적으로 작동
    device=-1
)

llm = HuggingFacePipeline(
    pipeline=qa_pipeline,
    model_kwargs={"max_new_tokens": 100, "temperature": 0.7}
)

# ------------------------------
# ✅ LangChain RAG Chain 구성
# ------------------------------
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)

# ------------------------------
# ✅ FastAPI RAG 엔드포인트
# ------------------------------
@app.post("/rag")
def ask_rag(query: RAGQuery):
    result = rag_chain.invoke({"query": query.query})
    return {
        "query": query.query,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }


from langchain.prompts import PromptTemplate

# 프롬프트 템플릿
prompt = PromptTemplate.from_template("안녕하세요, {name}님. 무엇을 도와드릴까요?")
llm_chain = prompt | llm  # Pipe 방식으로 연결

@app.post("/llmchain")
def ask_llmchain(query: LLMChainQuery):
    result = llm_chain.invoke({"name": query.name})
    return {"message": result}



# ✅ Step 1: GroqAgentLLM 정의
from langchain_core.language_models.llms import LLM
from typing import ClassVar

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

@app.post("/agent")
def ask_agent(query: AgentQuery):
    result = agent_executor.invoke({"input": query.input})
    return {"result": result["output"]}


# main.py
from langgraph_flow import langgraph_graph

@app.post("/langgraph")
def run_langgraph(q: Question):
    return langgraph_graph.invoke({"question": q.prompt})



from advanced_langgraph_flow import rag_graph

@app.post("/adv_rag")
def run_advanced_rag(q: AdvRAGQuery):
    result = rag_graph.invoke({"query": q.query})
    return {
        "query": q.query,
        "answer": result["answer"],
        "docs": result.get("reranked", [])
    }

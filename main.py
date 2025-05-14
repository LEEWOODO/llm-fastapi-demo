import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from abc import ABC, abstractmethod
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

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

class Question(BaseModel):
    prompt: str

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

# class HuggingFaceProvider(LLMProvider):
#     def chat(self, prompt: str) -> str:
#         url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
#         headers = {
#             "Authorization": f"Bearer {os.getenv('HF_API_KEY')}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "inputs": f"Answer this: {prompt}"
#         }
#
#         response = requests.post(url, headers=headers, json=payload)
#
#         if response.status_code != 200:
#             return f"[HuggingFace 오류] 응답코드 {response.status_code}: {response.text}"
#
#         try:
#             result = response.json()
#             if isinstance(result, list) and "generated_text" in result[0]:
#                 return result[0]["generated_text"]
#             elif isinstance(result, list) and "output" in result[0]:
#                 return result[0]["output"]
#             return str(result)
#         except Exception as e:
#             return f"[파싱 실패] {str(e)}\n본문: {response.text}"


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
from pydantic import BaseModel

# ------------------------------
# ✅ RAG 쿼리 모델 정의
# ------------------------------
class RAGQuery(BaseModel):
    query: str

# ------------------------------
# ✅ OpenSearch 기반 벡터 검색 세팅
# ------------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = OpenSearchVectorSearch(
    index_name="rag-index",
    embedding_function=embedding,
    opensearch_url="http://localhost:9200"
)

# ------------------------------
# ✅ HuggingFace LLM 세팅 (예: Falcon-7B)
# ------------------------------
# qa_pipeline = pipeline(
#     "text-generation",
#     model="tiiuae/falcon-7b-instruct",
#     device_map="auto"
# )

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
from langchain_core.runnables import RunnableSequence

class LLMChainQuery(BaseModel):
    name: str

# 프롬프트 템플릿
prompt = PromptTemplate.from_template("안녕하세요, {name}님. 무엇을 도와드릴까요?")
llm_chain = prompt | llm  # Pipe 방식으로 연결

@app.post("/llmchain")
def ask_llmchain(query: LLMChainQuery):
    result = llm_chain.invoke({"name": query.name})
    return {"message": result}



# ✅ Step 1: GroqAgentLLM 정의
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
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

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool

# 계산기 도구 구성
tools = [
    Tool(
        name="Calculator",
        func=PythonREPLTool().run,
        description="수학 계산을 정확히 수행하는 도구입니다."
    )
]


from typing import Union
from langchain.agents.agent import AgentOutputParser
from langchain.schema.agent import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
import re

class SafeFinalAnswerParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )

        # ✅ fallback: LLM이 Action: Final Answer 로 착각했을 경우도 처리
        if '"action": "Final Answer"' in text:
            match = re.search(r'"action_input":\s*"(.*?)"', text)
            if match:
                return AgentFinish(
                    return_values={"output": match.group(1)},
                    log=text
                )

        raise OutputParserException(f"파싱 실패: Final Answer 포맷이 아닙니다\n{text}")



# Groq 기반 LLM 사용
llm = GroqAgentLLM()

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=False,
    agent_kwargs={
        "system_message": (
            "다음 규칙을 반드시 지키세요:\n"
            "1. Thought: 어떤 행동을 해야 하는지 작성\n"
            "2. Action: 반드시 'Calculator' 로만 작성\n"
            "3. Action Input: 반드시 수식만 (예: 3 * 400). '×', '곱하기' 금지\n"
            "4. Observation: 계산 결과\n"
            "5. Final Answer: 반드시 'Final Answer: 1200' 형식으로 마무리\n\n"
            "**절대 'Action: Final Answer' 형식은 쓰지 마세요!**\n"
            "**반드시 'Final Answer: ...' 형식으로 출력하세요!**"
        ),
        "output_parser": SafeFinalAnswerParser()
    }
)



class AgentQuery(BaseModel):
    input: str

@app.post("/agent")
def ask_agent(query: AgentQuery):
    result = agent_executor.invoke({"input": query.input})
    return {"result": result["output"]}





# main.py
from langgraph_flow import langgraph_graph

@app.post("/langgraph")
def run_langgraph(q: Question):
    return langgraph_graph.invoke({"question": q.prompt})
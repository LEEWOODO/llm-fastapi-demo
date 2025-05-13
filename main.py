import os
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

# ------------------------------
# ✅ GPT API 엔드포인트 (자동 전환)
# ------------------------------

@app.post("/chat")
def ask_chat(question: Question):
    try:
        provider = OpenAIProvider()  # openai 사용 가능한 경우
        return {"answer": provider.chat(question.prompt)}
    except Exception:
        provider = GroqProvider()  # fallback: groq 사용
        return {"answer": provider.chat(question.prompt)}
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
app = FastAPI()

from routes import items, rag, llmchain, agent, langgraph, advanced_rag

app.include_router(items.router)
app.include_router(rag.router)
app.include_router(llmchain.router)
app.include_router(agent.router)
app.include_router(langgraph.router)
app.include_router(advanced_rag.router)

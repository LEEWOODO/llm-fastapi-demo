from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
app = FastAPI()

from routes.agent import router as agent_router
from routes.items import router as items_router
from routes.langgraph import router as langgraph_router
from routes.multi_agent import router as multi_agent_router
from rag.rag import router as rag_router
from rag.llmchain import router as llmchain_router
from rag.rag_advanced_pipeline import router as advanced_rag_router

app.include_router(items_router)
app.include_router(rag_router)
app.include_router(llmchain_router)
app.include_router(agent_router)
app.include_router(langgraph_router)
app.include_router(advanced_rag_router)
app.include_router(multi_agent_router)

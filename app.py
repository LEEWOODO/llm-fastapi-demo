from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "우도 AI 서비스를 향해 첫걸음!!"}

from dotenv import load_dotenv
import os

# Load env variables first!
load_dotenv()

from fastapi import FastAPI
from app.api import ask

app = FastAPI(
    title="YKS AI RAG",
    description="YKS students RAG based question solving and topic explanation assistant.",
    version="1.0.0"
)

app.include_router(ask.router)

@app.get("/")
def root():
    return {"message": "Welcome to YKS AI RAG API. Use the /ask endpoint."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

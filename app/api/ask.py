from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from app.core.vectorstore import VectorStore
from app.core.embeddings import EmbeddingModel
from app.core.llm import LLMClient

router = APIRouter()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

# Global instances (lazy loading could be better but keeping it simple)
# We initialize them at module level or via dependency injection.
# For simplicity, we'll instantiate them here but load index on startup in main if needed,
# or just lazily here.

vector_store = VectorStore()
# Verify index exists
if not vector_store.load():
    print("WARNING: Index not found. Please run ingest/ingest_documents.py first.")

embedding_model = EmbeddingModel()
llm_client = None
try:
    llm_client = LLMClient()
except Exception as e:
    print(f"WARNING: LLM Client could not be initialized (API Key might be missing): {e}")

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not llm_client:
        raise HTTPException(status_code=500, detail="LLM Client could not be initialized. Check API Key.")
    
    if not vector_store.index:
         # Try loading again just in case
        if not vector_store.load():
            raise HTTPException(status_code=500, detail="Vector index not found. Please upload documents and run the ingest script.")

    # 1. Embed question
    query_embedding = embedding_model.get_query_embedding(request.question)
    
    # 2. Retrieve top-k chunks
    results = vector_store.search(query_embedding, top_k=5)
    
    if not results:
        return AskResponse(answer="No relevant documents found.", sources=[])

    # 3. Construct context
    context = ""
    sources = set()
    for metadata, score in results:
        context += f"--- Source: {metadata['source']} ---\n{metadata['content']}\n\n"
        sources.add(metadata['source'])
            
    # 4. Generate answer
    answer = llm_client.generate_answer(context, request.question)
    
    return AskResponse(answer=answer, sources=list(sources))

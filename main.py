"""
main.py
FastAPI entry point for the Enterprise RAG System API.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional

from src.pipeline.rag_chain import EnterpriseRAGChain
from src.observability.llm_monitor import LLMMonitor

app = FastAPI(
    title="Enterprise RAG API",
    description="Production RAG system with HIPAA compliance and LLM observability",
    version="1.0.0",
)

security = HTTPBearer()
monitor = LLMMonitor()
rag = EnterpriseRAGChain(index_path="faiss_index/")


class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list
    latency_ms: float
    model: str


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "your-secure-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, token: str = Depends(verify_token)):
    """Submit a question to the RAG system."""
    result = rag.query(question=request.question, user_id=request.user_id)
    monitor.log(
        query=request.question,
        response=result["answer"],
        latency_ms=result["latency_ms"],
        token_count=len(result["answer"].split()),
    )
    return result


@app.get("/metrics")
def get_metrics(token: str = Depends(verify_token)):
    """Get LLM observability metrics."""
    return monitor.get_metrics()


@app.get("/drift")
def get_drift(token: str = Depends(verify_token)):
    """Check for model drift against baseline."""
    return monitor.drift_report()


@app.get("/health")
def health():
    return {"status": "healthy", "index_docs": rag.vectorstore.index.ntotal}

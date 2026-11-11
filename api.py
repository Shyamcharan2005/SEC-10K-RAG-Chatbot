from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

import rag


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=10)


class Source(BaseModel):
    company: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_seconds: float
    generation_seconds: float


class HealthResponse(BaseModel):
    status: str
    chunks: int
    companies: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = rag.load_pipeline()
    yield


app = FastAPI(title="SEC 10-K RAG API", version="1.0.0", lifespan=lifespan)


def get_pipeline(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not loaded yet")
    return pipeline


@app.get("/health", response_model=HealthResponse)
def health(pipeline=Depends(get_pipeline)):
    return HealthResponse(status="ok", chunks=len(pipeline.chunks), companies=sorted(set(pipeline.companies)))


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, pipeline=Depends(get_pipeline)):
    try:
        result = pipeline.answer(body.question, top_k=body.top_k)
    except rag.GenerationError:
        raise HTTPException(status_code=503, detail="The language model is temporarily unavailable")
    return QueryResponse(
        answer=result.answer,
        sources=[Source(company=p.company, text=p.text) for p in result.passages],
        retrieval_seconds=round(result.retrieval_seconds, 3),
        generation_seconds=round(result.generation_seconds, 3),
    )
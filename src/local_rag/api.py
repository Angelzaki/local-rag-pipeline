from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from local_rag.pipeline import RagPipeline

app = FastAPI(title="local-rag-pipeline", version="0.1.0")
pipeline = RagPipeline()


class IngestRequest(BaseModel):
    docs_dir: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=2)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": len(pipeline.index.chunks),
        "engine": "ollama" if pipeline.settings.use_ollama else "heuristic",
    }


@app.post("/ingest")
def ingest_docs(body: IngestRequest) -> dict:
    docs_dir = Path(body.docs_dir) if body.docs_dir else None
    if docs_dir and not docs_dir.exists():
        raise HTTPException(status_code=400, detail="docs_dir no existe")
    count = pipeline.build_index(docs_dir)
    return {"chunks": count}


@app.post("/ask")
def ask(body: AskRequest) -> dict:
    if not pipeline.index.chunks:
        pipeline.build_index()
    result = pipeline.ask(body.question)
    return {
        "question": result.question,
        "answer": result.answer,
        "citations": result.citations,
        "scores": [round(s, 4) for s in result.scores],
        "abstained": result.abstained,
        "engine": result.engine,
        "guard_blocked": result.guard_blocked,
        "guard_score": result.guard_score,
    }

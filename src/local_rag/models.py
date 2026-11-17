from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    path: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class AskResult:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    abstained: bool = False
    engine: str = "heuristic"
    guard_blocked: bool = False
    guard_score: float | None = None

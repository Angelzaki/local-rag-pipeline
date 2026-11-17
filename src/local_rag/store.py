from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from local_rag.models import Chunk, Hit

TOKEN = re.compile(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text)]


class LocalIndex:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: list[Chunk] = chunks or []
        self._rebuild()

    def _rebuild(self) -> None:
        tokenized = [tokenize(c.text) for c in self.chunks]
        df: Counter[str] = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        n = len(tokenized) or 1
        self.idf = {t: math.log((n + 1) / (f + 1)) + 1.0 for t, f in df.items()}
        self.vectors: list[dict[str, float]] = []
        for tokens in tokenized:
            tf = Counter(tokens)
            length = len(tokens) or 1
            self.vectors.append({t: (c / length) * self.idf[t] for t, c in tf.items()})

    def replace(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._rebuild()

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q_tokens = tokenize(query)
        tf = Counter(q_tokens)
        length = len(q_tokens) or 1
        q_vec = {t: (c / length) * self.idf.get(t, 0.0) for t, c in tf.items()}
        scored = [
            Hit(chunk=chunk, score=self._cosine(q_vec, vec))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "start": c.start,
                "end": c.end,
            }
            for c in self.chunks
        ]
        (index_dir / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path) -> "LocalIndex":
        path = index_dir / "chunks.json"
        if not path.exists():
            return cls([])
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                start=item["start"],
                end=item["end"],
            )
            for item in raw
        ]
        return cls(chunks)

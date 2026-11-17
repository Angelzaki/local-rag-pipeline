from __future__ import annotations

from pathlib import Path

from local_rag.config import Settings, get_settings
from local_rag.generate import heuristic_answer, ollama_answer
from local_rag.ingest import ingest
from local_rag.models import AskResult
from local_rag.store import LocalIndex


class RagPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.index = LocalIndex.load(self.settings.index_dir)

    def build_index(self, docs_dir: Path | None = None) -> int:
        root = docs_dir or self.settings.docs_dir
        chunks = ingest(
            root,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        self.index.replace(chunks)
        self.index.save(self.settings.index_dir)
        return len(chunks)

    def ask(self, question: str) -> AskResult:
        hits = self.index.search(question, k=self.settings.top_k)
        if self.settings.use_ollama:
            return ollama_answer(
                question,
                hits,
                self.settings.min_score,
                self.settings.ollama_base_url,
                self.settings.ollama_model,
                enable_guard=self.settings.enable_guard,
                guard_threshold=self.settings.guard_threshold,
                min_overlap=self.settings.min_overlap,
            )
        return heuristic_answer(
            question,
            hits,
            self.settings.min_score,
            enable_guard=self.settings.enable_guard,
            guard_threshold=self.settings.guard_threshold,
            min_overlap=self.settings.min_overlap,
        )

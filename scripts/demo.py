from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_rag.config import Settings
from local_rag.pipeline import RagPipeline

QUESTIONS = [
    "¿Cuál es el descuento máximo en A1?",
    "¿Cuál es el límite comercial?",
    "¿A2 exige MFA?",
    "política de viajes interplanetarios 2099",
]


def main() -> int:
    settings = Settings(
        docs_dir=ROOT / "samples" / "docs",
        index_dir=ROOT / "data" / "index",
        use_ollama=False,
    )
    pipe = RagPipeline(settings)
    pipe.build_index()
    print(f"chunks={len(pipe.index.chunks)}")
    for q in QUESTIONS:
        result = pipe.ask(q)
        print(
            json.dumps(
                {
                    "q": q,
                    "answer": result.answer,
                    "citations": result.citations,
                    "abstained": result.abstained,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

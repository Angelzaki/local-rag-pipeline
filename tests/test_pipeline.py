from pathlib import Path

from local_rag.config import Settings
from local_rag.guard import check_faithfulness
from local_rag.pipeline import RagPipeline

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "samples" / "docs"


def test_ingest_and_ask(tmp_path):
    settings = Settings(
        docs_dir=DOCS,
        index_dir=tmp_path / "index",
        use_ollama=False,
        min_score=0.12,
        min_overlap=0.34,
        enable_guard=True,
    )
    pipe = RagPipeline(settings)
    n = pipe.build_index()
    assert n > 0

    result = pipe.ask("¿Cuál es el descuento máximo en A1?")
    assert not result.abstained
    assert "politica_a1" in result.citations
    assert "12" in result.answer
    assert result.guard_blocked is False
    assert result.guard_score is not None


def test_abstain_unknown(tmp_path):
    settings = Settings(
        docs_dir=DOCS,
        index_dir=tmp_path / "index",
        use_ollama=False,
        min_score=0.12,
        min_overlap=0.5,
    )
    pipe = RagPipeline(settings)
    pipe.build_index()
    result = pipe.ask("política de viajes interplanetarios 2099")
    assert result.abstained


def test_guard_blocks_unsupported():
    verdict = check_faithfulness(
        "A1 regala un viaje a la luna con descuento 90%",
        ["El descuento máximo autorizado en A1 es 12% sobre tarifa lista."],
    )
    assert verdict.allowed is False

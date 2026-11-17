from __future__ import annotations

import httpx

from local_rag.guard import check_faithfulness
from local_rag.models import AskResult, Hit
from local_rag.store import tokenize

ABSTAIN = "No cuento con evidencia suficiente en el corpus para responder."
GUARD_FALLBACK = "No puedo afirmar eso con la evidencia disponible en el contexto."
STOP = {
    "politica",
    "política",
    "cuales",
    "cuáles",
    "cual",
    "cuál",
    "como",
    "cómo",
    "que",
    "qué",
    "tiene",
    "tienen",
    "para",
    "sobre",
}


def _has_evidence(
    question: str,
    chunk_text: str,
    score: float,
    min_score: float,
    min_overlap: float = 0.34,
) -> bool:
    if score < min_score:
        return False
    content = [t for t in tokenize(question) if t not in STOP and len(t) > 3]
    if not content:
        return score >= max(min_score, 0.25)
    ctx = set(tokenize(chunk_text))
    overlap = sum(1 for t in content if t in ctx)
    return (overlap / len(content)) >= min_overlap


def _apply_guard(result: AskResult, contexts: list[str], enabled: bool, threshold: float) -> AskResult:
    if not enabled or result.abstained:
        return result
    verdict = check_faithfulness(result.answer, contexts, threshold=threshold)
    result.guard_score = verdict.score
    if not verdict.allowed:
        result.answer = GUARD_FALLBACK
        result.guard_blocked = True
        result.abstained = True
        result.citations = []
    return result


def heuristic_answer(
    question: str,
    hits: list[Hit],
    min_score: float,
    enable_guard: bool = True,
    guard_threshold: float = 0.55,
    min_overlap: float = 0.34,
) -> AskResult:
    if not hits or not _has_evidence(
        question, hits[0].chunk.text, hits[0].score, min_score, min_overlap
    ):
        return AskResult(
            question=question,
            answer=ABSTAIN,
            citations=[],
            scores=[h.score for h in hits],
            abstained=True,
            engine="heuristic",
        )

    best = hits[0].chunk.text
    sentences = [s.strip() for s in best.replace("\n", " ").split(".") if s.strip()]
    q_words = {w.lower() for w in question.split() if len(w) > 3}
    ranked = sorted(
        sentences,
        key=lambda s: sum(1 for w in q_words if w in s.lower()),
        reverse=True,
    )
    picked = ranked[:2] if ranked else [best[:220]]
    answer = ". ".join(picked).strip() + "."
    citations = list(dict.fromkeys(h.chunk.doc_id for h in hits if h.score >= min_score))
    if not citations:
        citations = [hits[0].chunk.doc_id]
    result = AskResult(
        question=question,
        answer=answer,
        citations=citations,
        scores=[h.score for h in hits],
        abstained=False,
        engine="heuristic",
    )
    contexts = [h.chunk.text for h in hits]
    return _apply_guard(result, contexts, enable_guard, guard_threshold)


def ollama_answer(
    question: str,
    hits: list[Hit],
    min_score: float,
    base_url: str,
    model: str,
    enable_guard: bool = True,
    guard_threshold: float = 0.55,
    min_overlap: float = 0.34,
) -> AskResult:
    if not hits or not _has_evidence(
        question, hits[0].chunk.text, hits[0].score, min_score, min_overlap
    ):
        return AskResult(
            question=question,
            answer=ABSTAIN,
            citations=[],
            scores=[h.score for h in hits],
            abstained=True,
            engine="ollama",
        )

    context = "\n\n".join(f"[{h.chunk.doc_id}] {h.chunk.text}" for h in hits)
    prompt = (
        "Responde solo con evidencia del contexto. "
        "Si no alcanza, di exactamente: "
        f"{ABSTAIN}\n\n"
        f"Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
    )
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{base_url.rstrip('/')}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
    except Exception:
        return heuristic_answer(
            question, hits, min_score, enable_guard, guard_threshold, min_overlap
        )

    abstained = ABSTAIN.lower() in text.lower() or not text
    citations = [] if abstained else list(dict.fromkeys(h.chunk.doc_id for h in hits))
    result = AskResult(
        question=question,
        answer=text or ABSTAIN,
        citations=citations,
        scores=[h.score for h in hits],
        abstained=abstained,
        engine="ollama",
    )
    return _apply_guard(result, [h.chunk.text for h in hits], enable_guard, guard_threshold)

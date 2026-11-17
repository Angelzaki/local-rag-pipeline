from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN = re.compile(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9]+")
STOP = {
    "el", "la", "los", "las", "de", "del", "un", "una", "y", "o", "en",
    "para", "con", "por", "que", "qué", "es", "son", "se", "al",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text) if t.lower() not in STOP]


@dataclass
class GuardVerdict:
    allowed: bool
    score: float
    unsupported_tokens: list[str]
    reason: str


def check_faithfulness(answer: str, contexts: list[str], threshold: float = 0.55) -> GuardVerdict:
    ans = tokenize(answer)
    if not ans:
        return GuardVerdict(True, 1.0, [], "empty")
    ctx: set[str] = set()
    for c in contexts:
        ctx.update(tokenize(c))
    unsupported = [t for t in ans if t not in ctx]
    score = (len(ans) - len(unsupported)) / len(ans)
    allowed = score >= threshold
    return GuardVerdict(
        allowed=allowed,
        score=round(score, 4),
        unsupported_tokens=unsupported[:20],
        reason="ok" if allowed else "unsupported_claims",
    )

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TUNER = WORKSPACE / "retrieval-tuner-local" / "results" / "retrieval_tune.json"
ENV_EXAMPLE = ROOT / ".env.example"
CONFIG = ROOT / "src" / "local_rag" / "config.py"


def main() -> int:
    if not TUNER.exists():
        print(f"no hay tuner result: {TUNER}")
        return 1
    best = json.loads(TUNER.read_text(encoding="utf-8"))["best"]
    top_k = best["top_k"]
    min_score = best["min_score"]

    env_text = ENV_EXAMPLE.read_text(encoding="utf-8")
    env_text = re.sub(r"TOP_K=.*", f"TOP_K={top_k}", env_text)
    env_text = re.sub(r"MIN_SCORE=.*", f"MIN_SCORE={min_score}", env_text)
    ENV_EXAMPLE.write_text(env_text, encoding="utf-8")

    cfg = CONFIG.read_text(encoding="utf-8")
    cfg = re.sub(r"top_k: int = .*", f"top_k: int = {top_k}", cfg)
    cfg = re.sub(r"min_score: float = .*", f"min_score: float = {min_score}", cfg)
    CONFIG.write_text(cfg, encoding="utf-8")

    print(json.dumps({"synced": True, "top_k": top_k, "min_score": min_score, "hit_rate": best["hit_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

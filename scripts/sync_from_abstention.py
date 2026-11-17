from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CALIB = WORKSPACE / "abstention-lab-local" / "results" / "abstention_calibration.json"
ENV_EXAMPLE = ROOT / ".env.example"
CONFIG = ROOT / "src" / "local_rag" / "config.py"


def main() -> int:
    if not CALIB.exists():
        print(f"no hay calibración: {CALIB}")
        return 1
    best = json.loads(CALIB.read_text(encoding="utf-8"))["best"]
    min_score = best["min_score"]
    min_overlap = best["min_overlap"]

    env_text = ENV_EXAMPLE.read_text(encoding="utf-8")
    if "MIN_OVERLAP=" not in env_text:
        env_text = env_text.rstrip() + f"\nMIN_OVERLAP={min_overlap}\n"
    else:
        env_text = re.sub(r"MIN_OVERLAP=.*", f"MIN_OVERLAP={min_overlap}", env_text)
    env_text = re.sub(r"MIN_SCORE=.*", f"MIN_SCORE={min_score}", env_text)
    ENV_EXAMPLE.write_text(env_text, encoding="utf-8")

    cfg = CONFIG.read_text(encoding="utf-8")
    cfg = re.sub(r"min_score: float = .*", f"min_score: float = {min_score}", cfg)
    if "min_overlap: float" not in cfg:
        cfg = cfg.replace(
            f"min_score: float = {min_score}",
            f"min_score: float = {min_score}\n    min_overlap: float = {min_overlap}",
        )
    else:
        cfg = re.sub(r"min_overlap: float = .*", f"min_overlap: float = {min_overlap}", cfg)
    CONFIG.write_text(cfg, encoding="utf-8")

    print(
        json.dumps(
            {
                "synced": True,
                "min_score": min_score,
                "min_overlap": min_overlap,
                "accuracy": best["accuracy"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

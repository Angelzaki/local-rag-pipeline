from __future__ import annotations

import argparse
import json

import uvicorn

from local_rag.pipeline import RagPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_p = sub.add_parser("ingest", help="indexar documentos locales")
    ingest_p.add_argument("--docs", default=None)

    ask_p = sub.add_parser("ask", help="preguntar al índice")
    ask_p.add_argument("question")

    serve_p = sub.add_parser("serve", help="API FastAPI local")
    serve_p.add_argument("--host", default=None)
    serve_p.add_argument("--port", type=int, default=None)

    args = parser.parse_args(argv)
    pipe = RagPipeline()

    if args.cmd == "ingest":
        from pathlib import Path

        n = pipe.build_index(Path(args.docs) if args.docs else None)
        print(json.dumps({"chunks": n}, ensure_ascii=False))
        return 0

    if args.cmd == "ask":
        if not pipe.index.chunks:
            pipe.build_index()
        result = pipe.ask(args.question)
        print(
            json.dumps(
                {
                    "answer": result.answer,
                    "citations": result.citations,
                    "abstained": result.abstained,
                    "engine": result.engine,
                    "scores": [round(s, 4) for s in result.scores],
                    "guard_blocked": result.guard_blocked,
                    "guard_score": result.guard_score,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.cmd == "serve":
        host = args.host or pipe.settings.host
        port = args.port or pipe.settings.port
        uvicorn.run("local_rag.api:app", host=host, port=port, reload=False)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

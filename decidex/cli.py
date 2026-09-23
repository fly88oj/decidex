"""Command line: serve the API or run a quick offline demo.

    python -m decidex serve --engine llm --model Qwen/Qwen3-4B --port 8600
    python -m decidex demo  --engine embedding
"""

from __future__ import annotations

import argparse
import time


def _build_engine_factory(kind, model, temperature, device=None, lora_path=None,
                          dtype=None):
    def factory():
        from decidex.engines import build_engine

        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if device is not None and kind != "stub":
            kwargs["device"] = device
        if lora_path and kind == "llm":
            kwargs["lora_path"] = lora_path
        if dtype and kind == "llm":
            kwargs["dtype"] = dtype
        engine = build_engine(kind, model=model, **kwargs)
        print(f"[decidex] engine={engine.name} model={engine.model_id} temperature={engine.temperature}")
        return engine

    return factory


def _serve(args) -> None:
    import uvicorn

    from decidex.server import create_app

    factory = _build_engine_factory(args.engine, args.model, args.temperature,
                                    args.device, args.lora, args.dtype)
    app = create_app(
        engine=None,
        engine_factory=factory,
        api_key=args.api_key,
    )
    print(f"[decidex] serving on http://{args.host}:{args.port} (POST /v1/systemone)")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


def _demo(args) -> None:
    from decidex.calib import confidence
    from decidex.server import evaluate

    engine = _build_engine_factory(args.engine, args.model, args.temperature,
                                    args.device, args.lora, args.dtype)()
    state = {
        "ticket": {
            "subject": "Duplicate charge",
            "messages": [
                {
                    "from": "customer",
                    "text": "I was charged twice for order A-104. "
                            "This is unacceptable, refund me NOW.",
                },
            ],
        },
        "refund_policy": "Duplicate charges are eligible for a full refund.",
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does `ticket.messages[0].text` request a refund?",
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer appear in `ticket.messages[0].text`?",
            "criteria": ["Calm and neutral", "Frustrated but civil", "Very angry, strong language"],
        },
    }
    started = time.perf_counter()
    answers = evaluate(engine, state, questions)
    elapsed = (time.perf_counter() - started) * 1000

    import json

    print(json.dumps(answers, indent=2, ensure_ascii=False))
    dept = answers["department"]
    print(f"\n[{engine.name}/{engine.model_id}] {len(questions)} questions in {elapsed:.0f} ms")
    print(f"department={dept['choice']} confidence={dept['confidence']} "
          f"(calibration check: formula confidence={confidence(list(dept['probabilities'].values())):.4f})")
    print(f"refund_requested.noul={answers['refund_requested']['noul']}")
    print(f"frustration.score={answers['frustration']['score']} / 2")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="decidex", description="Local Jev-like System One decision model")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("serve", "run the HTTP API"), ("demo", "run one evaluation offline")):
        # demo defaults to the dependency-free stub so the quickstart works
        # before any ML extras are installed; pass --engine embedding/llm for
        # real semantic output.
        sub_parser = sub.add_parser(name, help=help_text)
        sub_parser.add_argument("--engine", default="llm" if name == "serve" else "stub",
                                choices=["llm", "embedding", "stub"], help="scoring engine")
        sub_parser.add_argument("--model", default=None, help="model name/path (default: engine-specific)")
        sub_parser.add_argument("--temperature", type=float, default=None, help="softmax temperature")
        sub_parser.add_argument("--device", default=None,
                                help="torch device, e.g. cuda:1 (default: DECIDEX_DEVICE env, else cuda/cpu)")
        sub_parser.add_argument("--lora", default=None, dest="lora",
                                help="LoRA adapter path for the llm engine (default: DECIDEX_LORA env)")
        sub_parser.add_argument("--dtype", default=None,
                                choices=["auto", "int8", "int4"],
                                help="model load precision; int8/int4 fits 7-8B models in 16GB")

    serve_parser = sub.choices["serve"]
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8600)
    serve_parser.add_argument("--api-key", default=None,
                              help="require this Bearer token (default: DECIDEX_API_KEY env)")
    serve_parser.add_argument("--log-level", default="info")

    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args)
    else:
        _demo(args)


if __name__ == "__main__":
    main()

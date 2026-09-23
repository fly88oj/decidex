"""Benchmark runner: accuracy, calibration, and latency for an engine.

Usage:
    HF_HOME=... DECIDEX_DEVICE=cuda:1 python benchmarks/bench.py \
        [--model Qwen/Qwen3-4B] [--temperature 1.0] [--tag baseline] [--json out.json]

Measures:
- Noul accuracy (p_yes vs 0.5) and Brier score (calibration of the belief)
- Choice top-1 accuracy
- Score accuracy (|score - expected| <= 0.75) and MAE
- Reliability: average stated confidence (top probability) vs empirical
  accuracy — a well-calibrated model states ~its true hit rate
- Latency: cold and warm single-item p50/p95, and 10-question fan-out
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import all_items  # noqa: E402

from decidex.engines.llm_logits import LLMLogitsEngine  # noqa: E402
from decidex.render import render_state, render_text  # noqa: E402


def noul_option_texts(criteria: dict | None) -> list[str]:
    criteria = criteria or {}
    yes = f"yes - {render_text(criteria['true'])}" if criteria.get("true") else "yes"
    no = f"no - {render_text(criteria['false'])}" if criteria.get("false") else "no"
    return [yes, no]


def choice_option_texts(criteria: dict) -> list[str]:
    out = []
    for key, description in criteria.items():
        rendered = render_text(description) if description is not None else None
        out.append(f"{key} - {rendered}" if rendered is not None else str(key))
    return out


def item_job(item: dict) -> tuple[str, str, list[str]]:
    state_text = render_state(item["state"])
    instruction = render_text(item["instruction"])
    if item["kind"] == "noul":
        options = noul_option_texts(item["criteria"])
    elif item["kind"] == "choice":
        options = choice_option_texts(item["criteria"])
    else:
        options = [render_text(level) for level in item["criteria"]]
    return state_text, instruction, options


def run_accuracy(engine: LLMLogitsEngine) -> dict:
    items = all_items()
    jobs = [item_job(item) for item in items]

    t0 = time.perf_counter()
    distributions = []
    for job in jobs:  # one request per item, like SemIf's authored decisions
        distributions.append(engine.score(*job))
    eval_seconds = time.perf_counter() - t0

    noul_correct, noul_brier, choice_correct, choice_top_probs = [], [], [], []
    score_correct, score_mae = [], []
    for item, probs in zip(items, distributions):
        if item["kind"] == "noul":
            p_yes = probs[0]
            noul_correct.append(int(p_yes > 0.5) == item["expected"])
            noul_brier.append((p_yes - item["expected"]) ** 2)
        elif item["kind"] == "choice":
            keys = list(item["criteria"].keys())
            top = max(range(len(keys)), key=lambda i: probs[i])
            choice_correct.append(keys[top] == item["expected"])
            choice_top_probs.append(max(probs))
        else:
            expected = float(sum(i * p for i, p in enumerate(probs)))
            score_mae.append(abs(expected - item["expected"]))
            score_correct.append(abs(expected - item["expected"]) <= 0.75)

    avg_top_prob = statistics.mean(choice_top_probs)
    choice_acc = statistics.mean(choice_correct)
    return {
        "items": len(items),
        "eval_seconds": round(eval_seconds, 2),
        "noul": {"n": len(noul_correct), "accuracy": round(statistics.mean(noul_correct), 4),
                 "brier": round(statistics.mean(noul_brier), 4)},
        "choice": {"n": len(choice_correct), "accuracy": round(choice_acc, 4),
                   "avg_top_prob": round(avg_top_prob, 4),
                   # calibration gap: stated confidence minus empirical accuracy
                   "gap": round(avg_top_prob - choice_acc, 4)},
        "score": {"n": len(score_correct),
                  "accuracy": round(statistics.mean(score_correct), 4),
                  "mae": round(statistics.mean(score_mae), 4)},
        "overall_accuracy": round(statistics.mean(
            noul_correct + choice_correct + score_correct), 4),
    }


def run_latency(engine: LLMLogitsEngine) -> dict:
    items = all_items()
    jobs = [item_job(item) for item in items]

    # warmup
    engine.score(*jobs[0])

    singles = []
    for job in jobs:
        t0 = time.perf_counter()
        engine.score(*job)
        singles.append((time.perf_counter() - t0) * 1000)
    singles.sort()

    # fan-out: 10 questions over one shared state, one request
    state, instruction, options = jobs[0]
    fanout_jobs = [(state, f"{instruction} (variant {i})", options) for i in range(10)]
    fanout = []
    for _ in range(5):
        t0 = time.perf_counter()
        engine.score_batch(fanout_jobs)
        fanout.append((time.perf_counter() - t0) * 1000)

    # long-state fan-out: the realistic prefix-reuse case (many questions over
    # one document). ~1.5k-token synthetic policy document.
    long_state = (
        "COMPANY SUPPORT POLICY DOCUMENT (v4.2).\n"
        + "".join(
            f"Section {i}. Customers experiencing issues with billing, access, or "
            f"shipping should first consult the FAQ at section {max(0, i - 1)}. "
            f"Refunds for order {1000 + i} are processed within {3 + i % 5} business "
            f"days when the item is unopened. Escalation to tier {1 + i % 3} support "
            f"is authorized only after two failed resolution attempts. Agents must "
            f"log every interaction in the CRM with tag POL-{i} before closing.\n"
            for i in range(40)
        )
    )
    long_fanout = [
        (long_state, f"Does section {i * 4} mention escalation rules?", ["yes", "no"])
        for i in range(10)
    ]
    long_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        engine.score_batch(long_fanout)
        long_ms.append((time.perf_counter() - t0) * 1000)

    def p(values: list[float], q: float) -> float:
        return round(values[min(len(values) - 1, int(q * len(values)))], 1)

    return {
        "single_ms_p50": p(singles, 0.5),
        "single_ms_p95": p(singles, 0.95),
        "fanout10_ms_median": round(statistics.median(fanout), 1),
        "fanout10_ms_min": round(min(fanout), 1),
        "longstate_fanout10_ms_median": round(statistics.median(long_ms), 1),
        "longstate_fanout10_ms_min": round(min(long_ms), 1),
        "longstate_tokens": len(engine.tokenizer.encode(long_state)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--json", default=None, help="append result to this JSON file")
    args = parser.parse_args()

    t0 = time.perf_counter()
    engine = LLMLogitsEngine(model_name=args.model, device=None,
                             temperature=args.temperature if args.temperature is not None else 1.0)
    load_seconds = round(time.perf_counter() - t0, 1)

    results = {
        "tag": args.tag,
        "model": engine.model_id,
        "temperature": engine.temperature,
        "load_seconds": load_seconds,
        "accuracy": run_accuracy(engine),
        "latency": run_latency(engine),
    }
    print(json.dumps(results, indent=2))

    if args.json:
        path = Path(args.json)
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing.append(results)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"\nappended to {path}")


if __name__ == "__main__":
    main()

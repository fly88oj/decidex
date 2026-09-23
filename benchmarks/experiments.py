"""Accuracy/calibration experiment matrix for the LLM engine.

Runs the authored benchmark under each prompt variant, with and without
option-order ensembling, and fits the softmax temperature on captured
τ=1.0 probabilities (argmax is temperature-invariant, so fitting improves
calibration, not accuracy — the numbers show exactly that).

    HF_HOME=... DECIDEX_DEVICE=cuda:1 python benchmarks/experiments.py \
        [--model Qwen/Qwen3-4B]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench import choice_option_texts, noul_option_texts  # noqa: E402
from dataset import all_items  # noqa: E402

from decidex.engines.llm_logits import LLMLogitsEngine  # noqa: E402
from decidex.render import render_state, render_text  # noqa: E402


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


def capture(engine: LLMLogitsEngine) -> list[tuple[dict, list[float]]]:
    items = all_items()
    results = []
    for item in items:
        probs = engine.score(*item_job(item))
        results.append((item, probs))
    return results


def metrics(captured: list[tuple[dict, list[float]]], tau: float = 1.0) -> dict:
    noul_correct, noul_brier, choice_correct, choice_top, choice_nll = [], [], [], [], []
    score_correct, score_mae = [], []
    for item, probs in captured:
        if item["kind"] == "noul":
            scores = [math.log(max(p, 1e-12)) for p in probs]
            p_yes = _softmax_tau(scores, tau)[0]
            noul_correct.append(int(p_yes > 0.5) == item["expected"])
            noul_brier.append((p_yes - item["expected"]) ** 2)
        elif item["kind"] == "choice":
            keys = list(item["criteria"].keys())
            expected_idx = keys.index(item["expected"])
            adjusted = _softmax_tau([math.log(max(p, 1e-12)) for p in probs], tau)
            top = max(range(len(keys)), key=lambda i: adjusted[i])
            choice_correct.append(top == expected_idx)
            choice_top.append(max(adjusted))
            choice_nll.append(-math.log(max(adjusted[expected_idx], 1e-12)))
        else:
            expected = float(sum(i * p for i, p in enumerate(probs)))
            score_mae.append(abs(expected - item["expected"]))
            score_correct.append(abs(expected - item["expected"]) <= 0.75)

    def mean(xs): return sum(xs) / len(xs)

    return {
        "noul_acc": round(mean(noul_correct), 3),
        "noul_brier": round(mean(noul_brier), 4),
        "choice_acc": round(mean(choice_correct), 3),
        "choice_nll": round(mean(choice_nll), 4),
        "choice_gap": round(mean(choice_top) - mean(choice_correct), 4),
        "score_acc": round(mean(score_correct), 3),
        "score_mae": round(mean(score_mae), 3),
    }


def _softmax_tau(scores: list[float], tau: float) -> list[float]:
    scaled = [s / tau for s in scores]
    peak = max(scaled)
    exps = [math.exp(s - peak) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def fit_temperature(captured: list[tuple[dict, list[float]]]) -> tuple[float, float]:
    """Grid-search tau minimizing NLL over noul + choice items."""
    best_tau, best_nll = 1.0, float("inf")
    tau = 0.70
    while tau <= 3.05:
        nll = metrics(captured, tau)["choice_nll"] * len(captured)
        # include noul NLL
        noul_nll = 0.0
        for item, probs in captured:
            if item["kind"] == "noul":
                p_yes = _softmax_tau([math.log(max(p, 1e-12)) for p in probs], tau)[0]
                y = item["expected"]
                noul_nll += -(y * math.log(max(p_yes, 1e-12)) + (1 - y) * math.log(max(1 - p_yes, 1e-12)))
        total = nll + noul_nll
        if total < best_nll:
            best_nll, best_tau = total, tau
        tau += 0.05
    return round(best_tau, 2), best_nll


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    engine = LLMLogitsEngine(model_name=args.model, device=None, temperature=1.0)
    print(f"model={engine.model_id} items={len(all_items())}\n")

    print(f"{'variant':8} {'ens':>3} {'noul_acc':>8} {'brier':>7} {'cho_acc':>7} "
          f"{'cho_gap':>7} {'score':>6} {'fit_tau':>7} {'brier@tau':>9}")
    for variant in ("plain", "strict", "chat"):
        engine.prompt_variant = variant
        for ensemble in (1, 3):
            engine.ensemble_rounds = ensemble
            t0 = time.perf_counter()
            captured = capture(engine)
            elapsed = time.perf_counter() - t0
            m = metrics(captured)
            tau, _ = fit_temperature(captured)
            m_fitted = metrics(captured, tau)
            print(f"{variant:8} {ensemble:>3} {m['noul_acc']:>8} {m['noul_brier']:>7} "
                  f"{m['choice_acc']:>7} {m['choice_gap']:>7} {m['score_acc']:>6} "
                  f"{tau:>7} {m_fitted['noul_brier']:>9}   [{elapsed:.0f}s]")
    engine.prompt_variant, engine.ensemble_rounds = "plain", 1


if __name__ == "__main__":
    main()

"""Active-learning distillation: mine samples where local and official disagree.

Generates fresh states/questions (new seed, disjoint from everything before),
runs BOTH the local engine and the official API, and keeps:
  - every disagreement (decision flip, or probability distance over threshold)
  - a small fraction of agreements (guards against forgetting aligned behavior)

Output feeds distill_train.py alongside the base dataset, upweighted.

    HF_HOME=... DECIDEX_DEVICE=cuda:1 python benchmarks/distill_active.py \
        [--states 500] [--keep-agree-frac 0.2]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from distill_generate import (  # noqa: E402
    build_questions,
    build_states,
    fetch,
    option_texts,
    target_distribution,
)

from decidex.engines.llm_logits import LETTERS, LLMLogitsEngine  # noqa: E402
from decidex.render import render_state, render_text  # noqa: E402

# disagreement thresholds per kind
DISAGREE_PROB = 0.30   # |p_local - p_official| beyond this = disagreement
DISAGREE_JS = 0.10     # distribution distance for choice/score
DISAGREE_SCORE = 0.5   # |score_local - score_official|


def js(p: list[float], q: list[float]) -> float:
    import math

    m = [(a + b) / 2 for a, b in zip(p, q)]
    eps = 1e-12

    def kl(x, y):
        return sum(a * math.log(a / max(b, eps), 2) for a, b in zip(x, y) if a > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def disagreement(kind: str, local: list[float], official: list[float]) -> tuple[bool, str]:
    if kind == "noul":
        lp, op = local[0], official[0]
        if (lp > 0.5) != (op > 0.5):
            return True, "decision-flip"
        if abs(lp - op) > DISAGREE_PROB:
            return True, f"prob-gap {abs(lp-op):.2f}"
        return False, ""
    if kind == "choice":
        if local.index(max(local)) != official.index(max(official)):
            return True, "top1-flip"
        if js(local, official) > DISAGREE_JS:
            return True, f"js {js(local, official):.2f}"
        return False, ""
    # score
    if local.index(max(local)) != official.index(max(official)):
        return True, "modal-flip"
    ls = sum(i * p for i, p in enumerate(local))
    os_ = sum(i * p for i, p in enumerate(official))
    if abs(ls - os_) > DISAGREE_SCORE:
        return True, f"score-gap {abs(ls-os_):.2f}"
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=500)
    parser.add_argument("--seed", type=int, default=333777)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--keep-agree-frac", type=float, default=0.2)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--out", default=str(BENCH / "distill_dataset_active.jsonl"))
    args = parser.parse_args()

    import os

    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        env_file = BENCH.parents[0] / ".env"
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip()

    rng = random.Random(args.seed)
    states = build_states(rng, args.states)
    payloads = [(state, build_questions(rng, state)) for state in states]
    total_questions = sum(len(q) for _s, q in payloads)
    print(f"mining {args.states} states / {total_questions} questions")

    engine = LLMLogitsEngine(model_name=args.model, dtype=args.dtype,
                             lora_path=args.lora,
                             device=os.environ.get("DECIDEX_DEVICE", "cuda:1"))
    print("local engine ready; collecting official answers ...")

    official_answers: dict[int, dict] = {}

    with httpx.Client() as client:
        def run(item):
            idx, (state, questions) = item
            return idx, fetch(client, "https://openrouter.ai", key,
                              "~typesafe/jev-latest", state, questions)

        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for idx, (status, body) in pool.map(run, enumerate(payloads)):
                done += 1
                if done % 100 == 0:
                    print(f"  official {done}/{len(payloads)}")
                if status == 200:
                    official_answers[idx] = body.get("answers", {})

    print("scoring locally + filtering ...")
    kept, agree_kept, disagree_total = [], 0, 0
    for idx, (state, questions) in enumerate(payloads):
        if idx not in official_answers:
            continue
        jobs, metas = [], []
        for qid, question in questions.items():
            jobs.append((render_state(state), render_text(question.get("instructions")),
                         option_texts(question)))
            metas.append((qid, question))
        local_probs = engine.score_batch(jobs)
        for (qid, question), local in zip(metas, local_probs):
            answer = official_answers[idx].get(qid)
            if not answer:
                continue
            target = target_distribution(question, answer)
            if not target or len(target) < 2:
                continue
            disagree, why = disagreement(question["type"], local, target)
            record = {
                "state": state, "kind": question["type"],
                "instructions": question.get("instructions"),
                "options": option_texts(question),
                "letters": list(LETTERS[: len(target)]),
                "target": target,
            }
            if disagree:
                disagree_total += 1
                record["why"] = why
                kept.append(record)
            elif rng.random() < args.keep_agree_frac:
                agree_kept += 1
                kept.append(record)

    rng.shuffle(kept)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for record in kept:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    kinds = {}
    for r in kept:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"kept {len(kept)} samples ({disagree_total} disagreements + "
          f"{agree_kept} agreements): {kinds}")
    print(f"disagreement rate: {disagree_total}/{total_questions} "
          f"= {disagree_total/total_questions:.1%}")


if __name__ == "__main__":
    main()

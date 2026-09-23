"""Fit Decidex's probability shape to the official API's answers.

Targets come from a prior real comparison run (comparison_raw.json): every
corpus question has an official probability. Two knobs are optimized:

1. Noul default option texts — the bare "yes"/"no" readout diverges from the
   official on ambiguous judgments; candidate phrasings are searched.
2. Softmax temperature — the official (RLCD) probabilities are softer than
   raw τ=1.0 logits. A single τ is fit per question type.

Fitting uses one half of the corpus, validation the other half, so the
reported gains are not just memorization.

    HF_HOME=... DECIDEX_DEVICE=cuda:1 python benchmarks/fit_to_official.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_corpus import build_corpus  # noqa: E402

from decidex.engines.llm_logits import LLMLogitsEngine  # noqa: E402
from decidex.render import render_state, render_text  # noqa: E402

RAW = Path(__file__).parent / "comparison_raw.json"

# Candidate default (no-criteria) noul option texts. Each pair replaces the
# bare yes/no; criteria-bearing questions keep their explicit texts.
NOUL_TEXT_VARIANTS = {
    "bare": ("yes", "no"),
    "holds": ("yes - the statement holds", "no - the statement does not hold"),
    "true_statement": ("yes - this is true of the state",
                       "no - this is not true of the state"),
    "applies": ("yes - the description applies", "no - the description does not apply"),
}


def question_jobs() -> list[dict]:
    """Flatten the corpus into per-question records with official targets."""
    data = json.loads(RAW.read_text(encoding="utf-8"))
    official = data["official"]
    records = []
    for entry in build_corpus():
        runs = official[entry["id"]]
        if runs[0]["status"] != 200:
            continue
        answers = runs[0]["body"]["answers"]
        for qid, question in entry["request"]["questions"].items():
            answer = answers.get(qid)
            if not answer:
                continue
            records.append({
                "rid": entry["id"], "qid": qid, "type": question["type"],
                "state": entry["request"]["state"], "question": question,
                "official": answer,
            })
    return records


def noul_options(question: dict, variant: tuple[str, str]) -> list[str]:
    criteria = question.get("criteria") or {}
    yes = f"yes - {render_text(criteria['true'])}" if criteria.get("true") else variant[0]
    no = f"no - {render_text(criteria['false'])}" if criteria.get("false") else variant[1]
    return [yes, no]


def choice_options(criteria: dict) -> list[str]:
    out = []
    for key, description in criteria.items():
        rendered = render_text(description) if description is not None else None
        out.append(f"{key} - {rendered}" if rendered is not None else str(key))
    return out


def capture_base(engine: LLMLogitsEngine, records: list[dict],
                 noul_variant: tuple[str, str], score_anchor: bool = False) -> dict[str, list[float]]:
    """Score every record once at τ=1 with the given noul texts.

    Returns {record_key: base probabilities}; temperature is applied later
    analytically (softmax(log(p)/τ) == softmax(logits/τ)).
    """
    jobs, keys = [], []
    for rec in records:
        state_text = render_state(rec["state"])
        instruction = render_text(rec["question"].get("instructions"))
        if rec["type"] == "noul":
            options = noul_options(rec["question"], noul_variant)
        elif rec["type"] == "choice":
            options = choice_options(rec["question"]["criteria"])
        else:
            options = [render_text(level) for level in rec["question"]["criteria"]]
            if score_anchor:
                options = [f"level {i}: {text}" for i, text in enumerate(options)]
        jobs.append((state_text, instruction, options))
        keys.append(f"{rec['rid']}.{rec['qid']}")
    distributions = []
    # one job per state group, batched by the engine
    by_state: dict[str, list] = {}
    for key, job in zip(keys, jobs):
        by_state.setdefault(job[0], []).append((key, job))
    for state_text, items in by_state.items():
        results = engine.score_batch([job for _k, job in items])
        for (key, _job), probs in zip(items, results):
            distributions.append((key, probs))
    return dict(distributions)


def apply_tau(probs: list[float], tau: float) -> list[float]:
    scores = [math.log(max(p, 1e-12)) for p in probs]
    peak = max(scores)
    exps = [math.exp((s - peak) / tau) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def fit_tau(pairs: list[tuple[list[float], list[float]]]) -> tuple[float, float]:
    """Grid-search τ minimizing mean |Δ| between adjusted and target probs."""
    best_tau, best_loss = 1.0, float("inf")
    tau = 0.8
    while tau <= 3.01:
        loss = _mean_delta(pairs, tau)
        if loss < best_loss:
            best_tau, best_loss = tau, loss
        tau += 0.05
    return round(best_tau, 2), round(best_loss, 4)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def platt_transform(p_yes: float, a: float, b: float) -> float:
    logit = math.log(max(p_yes, 1e-12) / max(1.0 - p_yes, 1e-12))
    return _sigmoid(a * logit + b)


def fit_platt(noul_pairs: list[tuple[float, float]]) -> tuple[tuple[float, float], float]:
    """Grid-search (a, b) for p' = sigmoid(a·logit(p) + b) on noul yes-probs.

    a < 1 shrinks overconfident extremes toward the official's softer values.
    """
    best, best_loss = (1.0, 0.0), float("inf")
    a = 0.30
    while a <= 1.51:
        b = -0.4
        while b <= 0.41:
            loss = sum(abs(platt_transform(p, a, b) - t)
                       for p, t in noul_pairs) / len(noul_pairs)
            if loss < best_loss:
                best, best_loss = (round(a, 2), round(b, 2)), loss
            b += 0.05
        a += 0.05
    return best, round(best_loss, 4)


def platt_loss(noul_pairs: list[tuple[float, float]], params: tuple[float, float]) -> float:
    a, b = params
    return round(sum(abs(platt_transform(p, a, b) - t)
                     for p, t in noul_pairs) / len(noul_pairs), 4)


def _mean_delta(pairs: list[tuple[list[float], list[float]]], tau: float) -> float:
    total = 0.0
    count = 0
    for base, target in pairs:
        adjusted = apply_tau(base, tau)
        total += sum(abs(a - t) for a, t in zip(adjusted, target)) / len(base)
        count += 1
    return total / count if count else float("inf")


def evaluate_noul(records, base: dict, official) -> dict:
    """Noul agreement at τ=1 vs at the fitted τ."""
    keys = [f"{r['rid']}.{r['qid']}" for r in records]
    pairs = [([base[k][0], base[k][1]], official_p(r)) for k, r in zip(keys, records)]
    tau1 = _mean_delta(pairs, 1.0)
    return pairs, tau1


def official_p(rec) -> list[float]:
    if rec["type"] == "noul":
        p = rec["official"]["noul"]
        return [p, 1.0 - p]
    if rec["type"] == "choice":
        keys = list(rec["question"]["criteria"].keys())
        return [rec["official"]["probabilities"].get(k, 0.0) for k in keys]
    levels = rec["question"]["criteria"]
    return [rec["official"]["probabilities"].get(str(i), 0.0) for i in range(len(levels))]


def main() -> None:
    records = question_jobs()
    nouls = [r for r in records if r["type"] == "noul"]
    choices = [r for r in records if r["type"] == "choice"]
    scores = [r for r in records if r["type"] == "score"]
    print(f"targets: {len(nouls)} noul, {len(choices)} choice, {len(scores)} score")

    # deterministic split: even/odd indices
    def split(rs):
        return rs[0::2], rs[1::2]

    engine = LLMLogitsEngine(model_name=None, device=None, temperature=1.0)

    # ---- 1. noul option-text variants --------------------------------------
    print("\n== noul default-text variants (τ=1) ==")
    best_variant, best_loss = "bare", float("inf")
    variant_scores = {}
    for name, variant in NOUL_TEXT_VARIANTS.items():
        base = capture_base(engine, records, variant)
        fit_half, _ = split(nouls)
        loss = _mean_delta(
            [([base[f"{r['rid']}.{r['qid']}"][0], base[f"{r['rid']}.{r['qid']}"][1]],
              official_p(r)) for r in fit_half], 1.0)
        variant_scores[name] = round(loss, 4)
        if loss < best_loss:
            best_variant, best_loss = name, loss
        print(f"  {name:16} fit-half MAE {loss:.4f}")
    print(f"  -> best variant: {best_variant}")

    # ---- 2. temperature fit per type (with the best noul texts) ------------
    base = capture_base(engine, records, NOUL_TEXT_VARIANTS[best_variant])
    print("\n== temperature fit (fit-half -> validate-half) ==")
    results = {}
    for label, rs in (("noul", nouls), ("choice", choices), ("score", scores),
                      ("global", records)):
        fit_half, valid_half = split(rs)
        fit_pairs = [(base[f"{r['rid']}.{r['qid']}"], official_p(r)) for r in fit_half]
        valid_pairs = [(base[f"{r['rid']}.{r['qid']}"], official_p(r)) for r in valid_half]
        tau, fit_loss = fit_tau(fit_pairs)
        valid_base = _mean_delta(valid_pairs, 1.0)
        valid_fit = _mean_delta(valid_pairs, tau)
        results[label] = {"tau": tau, "fit_loss": fit_loss,
                          "valid_tau1": round(valid_base, 4),
                          "valid_fitted": round(valid_fit, 4)}
        print(f"  {label:7} τ={tau:<5} fit {fit_loss:.4f} | "
              f"validate τ1 {valid_base:.4f} -> fitted {valid_fit:.4f}")

    # ---- 3. Platt scaling for noul ----------------------------------------
    print("\n== Platt scaling for noul (fit-half -> validate-half) ==")
    fit_half, valid_half = split(nouls)
    fit_pairs = [(base[f"{r['rid']}.{r['qid']}"][0], official_p(r)[0]) for r in fit_half]
    valid_pairs = [(base[f"{r['rid']}.{r['qid']}"][0], official_p(r)[0]) for r in valid_half]
    params, fit_loss = fit_platt(fit_pairs)
    valid_raw = platt_loss(valid_pairs, (1.0, 0.0))
    valid_platt = platt_loss(valid_pairs, params)
    print(f"  a={params[0]} b={params[1]} | fit {fit_loss:.4f} | "
          f"validate raw {valid_raw:.4f} -> platt {valid_platt:.4f}")
    results["noul_platt"] = {"a": params[0], "b": params[1],
                             "fit_loss": fit_loss, "valid_raw": valid_raw,
                             "valid_platt": valid_platt}

    # ---- 4. score anchored level texts ------------------------------------
    print("\n== score anchored-level texts (τ=1, fit-half) ==")
    anchored = capture_base(engine, records, NOUL_TEXT_VARIANTS[best_variant],
                            score_anchor=True)
    fit_scores, _ = split(scores)
    plain_loss = _mean_delta([(base[f"{r['rid']}.{r['qid']}"], official_p(r))
                              for r in fit_scores], 1.0)
    anchor_loss = _mean_delta([(anchored[f"{r['rid']}.{r['qid']}"], official_p(r))
                               for r in fit_scores], 1.0)
    print(f"  plain {plain_loss:.4f} vs anchored {anchor_loss:.4f}")
    results["score_anchor"] = {"plain": round(plain_loss, 4),
                               "anchored": round(anchor_loss, 4)}

    out = {"best_noul_variant": best_variant, "variant_scores": variant_scores,
           "temperature": results}
    Path(__file__).parent.joinpath("fit_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\nwritten to benchmarks/fit_results.json")


if __name__ == "__main__":
    main()

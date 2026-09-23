"""Compare the official Jev API and the local Decidex server on identical requests.

    # real comparison (needs a real key):
    TYPESAFE_API_KEY=sk-... python benchmarks/compare_official.py \
        --official-url https://api.typesafe.ai --local-url http://127.0.0.1:8600

    # harness self-test (both endpoints point at the local server; agreement
    # must be perfect, which proves the metrics code itself):
    python benchmarks/compare_official.py --self-test

Outputs benchmarks/comparison_raw.json (per-request answers from both sides)
and a printed markdown summary. Dimensions and their validity rationale are
documented in the header of the report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_corpus import build_corpus, build_invalid  # noqa: E402

MAX_ATTEMPTS = 5


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def post(client: httpx.Client, base: str, path: str, body: dict, headers: dict) -> httpx.Response:
    """POST with 429/529 retry + backoff (honors Retry-After)."""
    delay = 1.0
    for attempt in range(MAX_ATTEMPTS):
        response = client.post(base.rstrip("/") + path, json=body, headers=headers)
        if response.status_code in (429, 529) and attempt < MAX_ATTEMPTS - 1:
            retry_after = response.headers.get("retry-after")
            time.sleep(float(retry_after) if retry_after else delay)
            delay = min(delay * 2, 30.0)
            continue
        return response
    return response  # type: ignore[possibly-undefined]


def collect(base: str, api_key: str, corpus: list[dict], path: str = "/v1/systemone",
             model_rewrite: str | None = None) -> dict[str, list[dict]]:
    """Run every request (with repeats) against one endpoint.

    ``model_rewrite`` swaps the request's ``model`` field when it equals the
    corpus default ("jev-latest") — used to target the OpenRouter slug
    ("~typesafe/jev-latest") without touching deliberately-invalid models
    like the unknown-model probe.

    Returns {request_id: [{status, answers|error, usage, ms}, ...]}.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    out: dict[str, list[dict]] = {}
    with httpx.Client(timeout=180) as client:
        for entry in corpus:
            body = entry["request"]
            if model_rewrite and body.get("model") == "jev-latest":
                body = dict(body, model=model_rewrite)
            runs = []
            for _ in range(entry.get("repeat", 1)):
                t0 = time.perf_counter()
                response = post(client, base, path, body, headers)
                elapsed = round((time.perf_counter() - t0) * 1000)
                try:
                    payload = response.json()
                except Exception:
                    payload = {"detail": response.text[:200]}
                runs.append({"status": response.status_code, "ms": elapsed, "body": payload})
            out[entry["id"]] = runs
    return out


# ---------------------------------------------------------------------------
# metrics (pure functions, exercised by the self-test)
# ---------------------------------------------------------------------------

def js_divergence(p: list[float], q: list[float], base: float = 2.0) -> float:
    """Jensen-Shannon divergence in [0,1] (base 2) between aligned distributions."""
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return 0.5 * _kl(p, m, base) + 0.5 * _kl(q, m, base)


def _kl(p: list[float], q: list[float], base: float) -> float:
    eps = 1e-12
    return sum(a * math.log(a / max(b, eps), base) for a, b in zip(p, q) if a > 0)


def kendall_tau(order_a: list, order_b: list) -> float:
    """Kendall tau between two rankings given as item sequences (best first).

    Items may be any hashable values (option keys); both sequences must
    contain the same items.
    """
    n = len(order_a)
    if n < 2:
        return 1.0
    if set(order_a) != set(order_b):
        return float("nan")  # incomparable rankings (different option sets)
    rank_b = {item: i for i, item in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            # order_a ranks items[i] above items[j] (i < j)
            if rank_b[order_a[i]] < rank_b[order_a[j]]:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def ranked_keys(probs: dict[str, float]) -> list[str]:
    return sorted(probs, key=probs.get, reverse=True)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare(official: dict, local: dict, corpus: list[dict]) -> dict:
    per_type = {"choice": [], "noul": [], "score": []}
    conf_pairs: list[tuple[float, float]] = []
    usage_ratios: list[float] = []
    disagreements = []

    for entry in corpus:
        off_runs = official[entry["id"]]
        loc_runs = local[entry["id"]]
        off_first, loc_first = off_runs[0], loc_runs[0]
        if off_first["status"] != 200 or loc_first["status"] != 200:
            disagreements.append({"id": entry["id"],
                                  "reason": f"status off={off_first['status']} loc={loc_first['status']}"})
            continue
        off_ans, loc_ans = off_first["body"]["answers"], loc_first["body"]["answers"]
        for qid, question in entry["request"]["questions"].items():
            off_a, loc_a = off_ans.get(qid), loc_ans.get(qid)
            if not off_a or not loc_a:
                disagreements.append({"id": f"{entry['id']}.{qid}", "reason": "missing answer"})
                continue
            if question["type"] == "choice":
                keys = list(question["criteria"].keys())
                po = [off_a["probabilities"].get(k, 0.0) for k in keys]
                pl = [loc_a["probabilities"].get(k, 0.0) for k in keys]
                agree = off_a["choice"] == loc_a["choice"]
                per_type["choice"].append({
                    "top1": int(agree),
                    "js": js_divergence(po, pl),
                    "l1": sum(abs(a - b) for a, b in zip(po, pl)),
                    "top_gap": abs(max(po) - max(pl)),
                    "tau": kendall_tau(ranked_keys(off_a["probabilities"]),
                                       ranked_keys(loc_a["probabilities"])),
                })
                conf_pairs.append((off_a["confidence"], loc_a["confidence"]))
                if not agree:
                    disagreements.append({"id": f"{entry['id']}.{qid}",
                                          "reason": f"choice off={off_a['choice']} "
                                                    f"loc={loc_a['choice']}"})
            elif question["type"] == "noul":
                per_type["noul"].append({
                    "delta": abs(off_a["noul"] - loc_a["noul"]),
                    "decision": int((off_a["noul"] > 0.5) == (loc_a["noul"] > 0.5)),
                })
            else:
                per_type["score"].append({
                    "delta": abs(off_a["score"] - loc_a["score"]),
                    "modal": int(ranked_keys(off_a["probabilities"])[0]
                                 == ranked_keys(loc_a["probabilities"])[0]),
                    "js": js_divergence(
                        [off_a["probabilities"].get(str(i), 0.0)
                         for i in range(len(question["criteria"]))],
                        [loc_a["probabilities"].get(str(i), 0.0)
                         for i in range(len(question["criteria"]))]),
                })
                conf_pairs.append((off_a["confidence"], loc_a["confidence"]))

        uo = off_first["body"].get("usage", {}).get("input_tokens")
        ul = loc_first["body"].get("usage", {}).get("input_tokens")
        if uo and ul:
            usage_ratios.append(ul / uo)

    def agg_choice(rows):
        return {"n": len(rows),
                "top1_agreement": _rate(r["top1"] for r in rows),
                "js_divergence": _mean(r["js"] for r in rows),
                "l1_distance": _mean(r["l1"] for r in rows),
                "top1_prob_gap": _mean(r["top_gap"] for r in rows),
                "kendall_tau": _mean(r["tau"] for r in rows)}

    def agg_noul(rows):
        return {"n": len(rows),
                "prob_mae": _mean(r["delta"] for r in rows),
                "max_delta": max((r["delta"] for r in rows), default=0.0),
                "decision_agreement": _rate(r["decision"] for r in rows)}

    def agg_score(rows):
        return {"n": len(rows),
                "score_mae": _mean(r["delta"] for r in rows),
                "max_delta": max((r["delta"] for r in rows), default=0.0),
                "modal_level_agreement": _rate(r["modal"] for r in rows),
                "js_divergence": _mean(r["js"] for r in rows)}

    self_consistency = _self_consistency(official, local, corpus)
    invariance = _invariance(official, local, corpus)
    conf_x = [p for p, _ in conf_pairs]
    conf_y = [q for _, q in conf_pairs]
    return {
        "choice": agg_choice(per_type["choice"]),
        "noul": agg_noul(per_type["noul"]),
        "score": agg_score(per_type["score"]),
        "confidence": {"n": len(conf_pairs),
                       "gap_mae": _mean(abs(x - y) for x, y in conf_pairs),
                       "pearson_r": round(pearson(conf_x, conf_y), 4)},
        "usage_input_ratio_local_over_official": {
            "n": len(usage_ratios),
            "mean": _mean(usage_ratios), "min": min(usage_ratios, default=0.0),
            "max": max(usage_ratios, default=0.0)},
        "self_consistency": self_consistency,
        "question_count_invariance": invariance,
        "disagreements": disagreements,
    }


def _self_consistency(official: dict, local: dict, corpus: list[dict]) -> dict:
    """Same request sent twice: how much does each side drift from itself?"""
    def drift(runs_by_id):
        deltas, flips = [], 0
        count = 0
        for entry in corpus:
            if entry.get("repeat", 1) < 2:
                continue
            runs = runs_by_id[entry["id"]]
            if any(r["status"] != 200 for r in runs[:2]):
                continue
            a1, a2 = runs[0]["body"]["answers"], runs[1]["body"]["answers"]
            for qid, question in entry["request"]["questions"].items():
                x, y = a1.get(qid), a2.get(qid)
                if not x or not y:
                    continue
                count += 1
                if question["type"] == "choice":
                    flips += int(x["choice"] != y["choice"])
                    deltas.append(js_divergence(
                        [x["probabilities"].get(k, 0.0) for k in question["criteria"]],
                        [y["probabilities"].get(k, 0.0) for k in question["criteria"]]))
                elif question["type"] == "noul":
                    flips += int((x["noul"] > 0.5) != (y["noul"] > 0.5))
                    deltas.append(abs(x["noul"] - y["noul"]))
                else:
                    deltas.append(abs(x["score"] - y["score"]))
        return {"questions": count, "top1_flips": flips, "mean_drift": _mean(deltas)}

    return {"official": drift(official), "local": drift(local)}


def _invariance(official: dict, local: dict, corpus: list[dict]) -> dict:
    """2-question vs 10-question requests over the same state: per-side drift
    on the SHARED question (official semantics: adding questions must not
    change other answers)."""
    by_pair: dict[str, list] = {}
    for entry in corpus:
        if entry.get("pair"):
            by_pair.setdefault(entry["pair"], []).append(entry)
    result = {}
    for pair_id, entries in by_pair.items():
        short, long = (min(entries, key=lambda e: len(e["request"]["questions"])),
                       max(entries, key=lambda e: len(e["request"]["questions"])))
        shared = set(short["request"]["questions"]) & set(long["request"]["questions"])

        def side_drift(runs_by_id, label):
            a = runs_by_id[short["id"]][0]
            b = runs_by_id[long["id"]][0]
            if a["status"] != 200 or b["status"] != 200:
                return {"error": "non-200"}
            deltas = []
            for qid in shared:
                x, y = a["body"]["answers"].get(qid), b["body"]["answers"].get(qid)
                if not x or not y:
                    continue
                if "noul" in x:
                    deltas.append(abs(x["noul"] - y["noul"]))
                elif "choice" in x:
                    deltas.append(int(x["choice"] != y["choice"]))
                else:
                    deltas.append(abs(x["score"] - y["score"]))
            return {"mean_drift": _mean(deltas), "samples": len(deltas)}

        result[pair_id] = {"official": side_drift(official, pair_id),
                           "local": side_drift(local, pair_id)}
    return result


def compare_invalid(official: dict, local: dict, invalid: list[dict]) -> list[dict]:
    rows = []
    for entry in invalid:
        off = official[entry["id"]][0]
        loc = local[entry["id"]][0]
        off_loc = _first_loc(off["body"])
        loc_loc = _first_loc(loc["body"])
        rows.append({
            "id": entry["id"],
            "official_status": off["status"], "local_status": loc["status"],
            "status_match": off["status"] == loc["status"],
            "official_loc": off_loc, "local_loc": loc_loc,
            "loc_match": off_loc == loc_loc,
        })
    return rows


def _first_loc(body: dict) -> str | None:
    detail = body.get("detail")
    if isinstance(detail, list) and detail:
        loc = detail[0].get("loc") or []
        return "/".join(str(x) for x in loc[1:]) if len(loc) > 1 else None
    return None


def _mean(xs):
    xs = list(xs)
    return round(statistics.mean(xs), 4) if xs else None


def _rate(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


# ---------------------------------------------------------------------------
# report + main
# ---------------------------------------------------------------------------

def render_report(metrics: dict, invalid_rows: list[dict], lat_off: dict, lat_loc: dict) -> str:
    c, n, s = metrics["choice"], metrics["noul"], metrics["score"]
    lines = [
        "# Decidex vs official Jev API — comparison report",
        "",
        "Agreement is measured on identical requests (see compare_corpus.py).",
        "Agreement is NOT correctness: without ground truth, deltas describe",
        "similarity, not which side is right. Latency is reported per-side only",
        "(remote API vs local GPU are not comparable).",
        "",
        "## Answer agreement",
        "",
        "| Dimension | Value |",
        "|---|---|",
        f"| Choice top-1 agreement | {c['top1_agreement']} (n={c['n']}) |",
        f"| Choice distribution JS divergence (0=identical, 1=disjoint) | {c['js_divergence']} |",
        f"| Choice distribution L1 distance | {c['l1_distance']} |",
        f"| Choice winner-probability gap | {c['top1_prob_gap']} |",
        f"| Choice option-ordering Kendall tau | {c['kendall_tau']} |",
        f"| Noul probability MAE | {n['prob_mae']} (n={n['n']}) |",
        f"| Noul max |Δp| | {n['max_delta']} |",
        f"| Noul 0.5-threshold decision agreement | {n['decision_agreement']} |",
        f"| Score MAE | {s['score_mae']} (n={s['n']}) |",
        f"| Score max |Δ| | {s['max_delta']} |",
        f"| Score modal-level agreement | {s['modal_level_agreement']} |",
        f"| Score distribution JS divergence | {s['js_divergence']} |",
        f"| Confidence gap MAE | {metrics['confidence']['gap_mae']} (n={metrics['confidence']['n']}) |",
        f"| Confidence Pearson r | {metrics['confidence']['pearson_r']} |",
        "",
        "## Behavioral semantics",
        "",
        "| Probe | Official | Local |",
        "|---|---|---|",
    ]
    sc = metrics["self_consistency"]
    lines.append(f"| Self-consistency (same request twice) | flips {sc['official']['top1_flips']}"
                 f"/{sc['official']['questions']}, drift {sc['official']['mean_drift']}"
                 f" | flips {sc['local']['top1_flips']}/{sc['local']['questions']},"
                 f" drift {sc['local']['mean_drift']} |")
    for pair, res in metrics["question_count_invariance"].items():
        lines.append(f"| Question-count invariance ({pair}) | drift {res['official']} | {res['local']} |")
    u = metrics["usage_input_ratio_local_over_official"]
    lines += [
        "",
        "## Contract behaviors",
        "",
        f"- usage.input_tokens ratio (local/official): mean {u['mean']}, "
        f"range [{u['min']}, {u['max']}] (n={u['n']}) — how close our tokenizer "
        "count is to the official metering.",
        "",
        "| Invalid request | Official status | Local status | Official field | Local field | Match |",
        "|---|---|---|---|---|---|",
    ]
    for row in invalid_rows:
        lines.append(f"| {row['id']} | {row['official_status']} | {row['local_status']} "
                     f"| {row['official_loc']} | {row['local_loc']} "
                     f"| {'yes' if row['status_match'] and row['loc_match'] else 'NO'} |")
    lines += [
        "",
        "## Latency (per-side, not head-to-head)",
        "",
        f"- Official: p50 {lat_off.get('p50')} ms, p95 {lat_off.get('p95')} ms, n={lat_off.get('n')}",
        f"- Local:    p50 {lat_loc.get('p50')} ms, p95 {lat_loc.get('p95')} ms, n={lat_loc.get('n')}",
        "",
        f"## Disagreements ({len(metrics['disagreements'])})",
        "",
    ]
    for d in metrics["disagreements"][:40]:
        lines.append(f"- {d['id']}: {d['reason']}")
    if len(metrics["disagreements"]) > 40:
        lines.append(f"- … {len(metrics['disagreements']) - 40} more (see comparison_raw.json)")
    return "\n".join(lines)


def latency_summary(official: dict, local: dict, corpus: list[dict]) -> tuple[dict, dict]:
    def lat(runs_by_id):
        xs = sorted(r["ms"] for e in corpus for r in runs_by_id[e["id"]][:1])
        if not xs:
            return {}
        return {"p50": round(statistics.median(xs)), "p95": xs[int(len(xs) * 0.95)],
                "n": len(xs)}
    return lat(official), lat(local)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-url", default="https://openrouter.ai")
    parser.add_argument("--official-path", default="/api/alpha/decisions",
                        help="official endpoint path (OpenRouter decisions gateway)")
    parser.add_argument("--official-model", default="~typesafe/jev-latest",
                        help="model slug sent to the official side (rewrites the "
                             "corpus default 'jev-latest'; invalid probes keep "
                             "their deliberate values)")
    parser.add_argument("--official-key", default=None,
                        help="defaults to TYPESAFE_API_KEY env or .env in the project root")
    parser.add_argument("--local-url", default="http://127.0.0.1:8600")
    parser.add_argument("--local-key", default="local-dev")
    parser.add_argument("--self-test", action="store_true",
                        help="both endpoints = local server; agreement must be perfect")
    parser.add_argument("--out", default=str(Path(__file__).parent / "comparison_raw.json"))
    args = parser.parse_args()

    import os

    official_key = args.official_key or os.environ.get("TYPESAFE_API_KEY", "")
    if not official_key:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("TYPESAFE_API_KEY="):
                    official_key = line.split("=", 1)[1].strip()
                    break
    official_url = args.local_url if args.self_test else args.official_url
    official_path = "/v1/systemone" if args.self_test else args.official_path
    official_model = None if args.self_test else args.official_model
    if not args.self_test and not official_key:
        raise SystemExit("No official API key: pass --official-key or set TYPESAFE_API_KEY "
                         "(or run --self-test to validate the harness locally)")

    corpus = build_corpus()
    invalid = build_invalid()
    print(f"corpus: {len(corpus)} requests "
          f"({sum(len(e['request']['questions']) for e in corpus)} questions), "
          f"{len(invalid)} invalid probes")

    print("collecting official ...")
    official = collect(official_url, official_key, corpus,
                       path=official_path, model_rewrite=official_model)
    official.update(collect(official_url, official_key, invalid,
                            path=official_path, model_rewrite=official_model))
    print("collecting local ...")
    local = collect(args.local_url, args.local_key, corpus)
    local.update(collect(args.local_url, args.local_key, invalid))

    metrics = compare(official, local, corpus)
    invalid_rows = compare_invalid(official, local, invalid)
    lat_off, lat_loc = latency_summary(official, local, corpus)

    Path(args.out).write_text(json.dumps(
        {"official": official, "local": local, "metrics": metrics,
         "invalid": invalid_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(metrics, invalid_rows, lat_off, lat_loc)
    Path(args.out).with_suffix(".md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nraw: {args.out}\nreport: {Path(args.out).with_suffix('.md')}")

    if args.self_test:
        ok = (metrics["choice"]["top1_agreement"] == 1.0
              and metrics["noul"]["prob_mae"] == 0.0
              and metrics["score"]["score_mae"] == 0.0
              and all(r["status_match"] and r["loc_match"] for r in invalid_rows))
        print("\nSELF-TEST:", "PASS (harness measures perfect agreement correctly)"
              if ok else "FAIL — harness bug!")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

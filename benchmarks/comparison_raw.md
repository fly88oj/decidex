# Decidex vs official Jev API — comparison report

Agreement is measured on identical requests (see compare_corpus.py).
Agreement is NOT correctness: without ground truth, deltas describe
similarity, not which side is right. Latency is reported per-side only
(remote API vs local GPU are not comparable).

## Answer agreement

| Dimension | Value |
|---|---|
| Choice top-1 agreement | 1.0 (n=23) |
| Choice distribution JS divergence (0=identical, 1=disjoint) | 0.0025 |
| Choice distribution L1 distance | 0.0176 |
| Choice winner-probability gap | 0.0088 |
| Choice option-ordering Kendall tau | 0.5583 |
| Noul probability MAE | 0.1295 (n=52) |
| Noul max |Δp| | 0.8797 |
| Noul 0.5-threshold decision agreement | 0.8846 |
| Score MAE | 0.4149 (n=11) |
| Score max |Δ| | 2.4346000000000005 |
| Score modal-level agreement | 0.7273 |
| Score distribution JS divergence | 0.1441 |
| Confidence gap MAE | 0.0862 (n=34) |
| Confidence Pearson r | 0.3766 |

## Behavioral semantics

| Probe | Official | Local |
|---|---|---|
| Self-consistency (same request twice) | flips 0/3, drift 0.0 | flips 0/3, drift 0.0 |
| Question-count invariance (invariance) | drift {'mean_drift': 0, 'samples': 1} | {'mean_drift': 0, 'samples': 1} |

## Contract behaviors

- usage.input_tokens ratio (local/official): mean 0.143, range [0.056291390728476824, 0.8650662251655629] (n=53) — how close our tokenizer count is to the official metering.

| Invalid request | Official status | Local status | Official field | Local field | Match |
|---|---|---|---|---|---|
| err-missing-state | 400 | 422 | None | state | NO |
| err-state-type | 400 | 422 | None | state | NO |
| err-empty-questions | 400 | 422 | None | questions | NO |
| err-bad-type | 400 | 422 | None | questions/q/type | NO |
| err-choice-300 | 400 | 422 | None | questions/q/criteria | NO |
| err-noul-bad-key | 400 | 422 | None | questions/q/criteria | NO |
| err-unknown-model | 400 | 422 | None | model | NO |
| err-score-nonlist | 400 | 422 | None | questions/q/criteria | NO |

## Latency (per-side, not head-to-head)

- Official: p50 783 ms, p95 1309 ms, n=54
- Local:    p50 52 ms, p95 650 ms, n=54

## Disagreements (1)

- edge-no-instructions: status off=400 loc=200
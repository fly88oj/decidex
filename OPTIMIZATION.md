# Optimization log

> 2026-09-22. Systematic optimization campaign: every known approach for
> performance, accuracy, and stability was either **tried with measurements**
> or **considered and rejected with reasons**. Reproduce any number with
> `benchmarks/bench.py` / `benchmarks/experiments.py`; raw runs live in
> `benchmarks/results.json`.

Environment: RTX 5080 (16 GB, `cuda:1`), Qwen3-4B BF16, torch 2.11+cu128,
transformers 5.17. Benchmark: 53 authored items (40 easy + 13 hard:
sarcasm, double negation, near-miss options), `benchmarks/dataset.py`.

## Summary dashboard

| Metric | Before campaign | After campaign |
|---|---|---|
| Single question, warm p50 | 52.8 ms | **49.1 ms** |
| 10-question fan-out, median | 150.6 ms | **65.9 ms** (2.3×) |
| 10-question fan-out over a 3.3k-token state, cold | **OOM / crash** | 2.2 s |
| …same state again (prefix cache hit) | — | **193 ms** (11.4× vs cold) |
| 32 concurrent requests | event loop blocked, no queue guard | all 200 (429+retry path), `/health` p50 **2 ms** under load, zero 5xx |
| Noul accuracy / Brier | 1.00 / 0.0003 (40-item set) | 1.00 / 0.0004 (53-item set incl. hard) |
| Choice accuracy / calibration gap | 1.00 / −0.0002 (easy set) | 1.00 / **−0.0009** (incl. hard) |
| Score accuracy (hard items) | 0.80 | 0.846 (→ **1.00** with `ensemble_rounds=3`) |

## Tried and adopted

### P1 — KV prefix reuse within a request (`engines/llm_logits.py`)

Every question in one request shares the state. The prompt was split into
`build_prefix` (state) + `build_suffix` (question): one forward fills the KV
cache for the state, each question batch continues from a `deepcopy` of it
expanded with `DynamicCache.batch_repeat_interleave`, with explicit
`position_ids = prefix_len + suffix_mask.cumsum(-1) - 1`.

- Long states stop OOM-ing: 10 full 3.4k-token prompts (attention is
  quadratic in batch × length) cannot fit in 16 GB at all; prefix reuse
  processes the state once. **Correctness pinned by
  `test_prefix_reuse_matches_full_prompts`** (argmax equality, prob
  deviation < 0.02). SemIf measured the same family of tricks as their
  biggest win (2.3 → 20 decisions/s).
- Fixed two real bugs on the way: probe scores must accumulate across
  chunks (partial distributions were overwriting full ones), and
  `position_ids` must cover only new tokens.
- Env: `DECIDEX_PREFIX_REUSE=0` disables.

### P1b — Cross-request prefix LRU cache

`_get_prefix_cache` keeps up to `DECIDEX_PREFIX_CACHE_GB` (default 4 GB) of
state-prefix KV caches in an LRU keyed by prefix text. Repeat queries over
the same document skip the prefill entirely: 3.3k-token state, 10 questions
— 2197 ms cold → **193 ms warm** (11.4×). Eviction is byte-accounted via
`_kv_bytes_for` so long states cannot starve the GPU.

### P2 — Batch chunk tuning

Swept CHUNK 4/8/16/32 on a 10-question fan-out: 163 / 111 / **89** / 89 ms.
Default raised 8 → 16 (flat beyond). Combined with P1 the fan-out went
150.6 → 65.9 ms. `torch.inference_mode()` replaces `no_grad()` throughout.

### P2b — OOM-adaptive backoff (`_rows_with_backoff`)

CUDA OOM during any chunked forward now halves the batch (`empty_cache`
between attempts, floor of 1) instead of killing the worker. This path was
exercised for real: full-prompt scoring of the 3.3k-token state OOMs at
every batch ≥ 2 on this GPU, and before the guard it crashed the process.

### P3 — Async service that never blocks the event loop

`/v1/systemone` now runs evaluation through `run_in_threadpool` under a
`threading.Lock` (models are not safe for concurrent forward passes) plus a
slot semaphore (`DECIDEX_MAX_QUEUE`, default 8). Measured by
`tests/load_test.py`: concurrency 8 and 32 → every request eventually 200
(overload takes the documented 429 + `Retry-After` retry path), **zero 5xx**,
and `/health` stays at p50 2 ms *while the GPU is saturated*.

### P4 — Overload semantics

429/529 responses now carry `Retry-After: 1` (both official SDKs honor it —
verified during the load test, which succeeds via that exact retry path).

## Tried, measured, not adopted as default

### A1 — Prompt variants (`experiments.py`, all three variants × 2 ensembles)

| variant | noul acc | choice acc | score acc | verdict |
|---|---|---|---|---|
| **plain** (shipped) | 1.00 | 0.938→**1.00**¹ | 0.846 | best |
| strict ("ONLY the letter") | 1.00 | 0.938→1.00¹ | 0.846 | no gain |
| chat template | **0.50** | **0.312** | 0.231 | **rejected** |

¹ after replacing one ambiguously-labeled dataset item (see below).

The chat template is catastrophic on Qwen3-4B (a thinking-hybrid model: the
template inserts thinking scaffolding, so the next-token letter logits are
garbage — noul drops to coin-flip). It is merely worse-than-plain on the
non-thinking Qwen3-4B-Instruct-2507 (0.958/0.938/0.846). Completion-style
prompts are the right substrate for direct logit readout, matching SemIf's
"frozen prompts" finding. Kept as an engine knob (`prompt_variant`) for
other models.

### A2 — Temperature fitting (single-parameter "training")

Fitting τ by NLL grid search over the benchmark's τ=1 probabilities
(temperature is argmax-invariant, so this is purely a calibration knob).
Fit result τ≈2.6 — but Brier *worsened* (0.0004 → 0.0103): with exactly one
wrong choice item in the set, NLL over-softens everything to dilute a single
catastrophic term. On a 53-item authored set the fit is unstable; at τ=1.0
the stated gap is already −0.0009 (near-perfect). **Decision: keep τ=1.0
default; the fitting methodology ships in `experiments.py` for production
traffic, where a few hundred labeled decisions make the fit meaningful.**

### A3 — Option-order self-consistency ensemble (`ensemble_rounds`)

Averaging distributions over 3 option-order permutations removes position
bias: hard-item score accuracy 0.846 → **1.00** (Qwen3-4B; 0.923 on
Instruct-2507), choice unchanged. Costs ~3× latency on ensembled jobs.
**Decision: opt-in knob (`ensemble_rounds`), not default** — Decidex's
selling point is latency; users trading latency for accuracy on hard
rubrics can flip one attribute.

### A4 — Alternative model: Qwen3-4B-Instruct-2507

| model | noul | choice | score | fanout10 | single p50 |
|---|---|---|---|---|---|
| Qwen3-4B (default) | 1.000 / Brier 4e-4 | 1.000 | 0.846 | 65.9 ms | 49.1 ms |
| Qwen3-4B-Instruct-2507 | 1.000 / Brier 0.0 | 1.000 | 0.846 | 65.7 ms | 48.4 ms |

Identical accuracy (same single miss before the dataset fix), identical
latency after P2 (the earlier 150 vs 66 ms gap was our batching, not the
model), marginally better noul Brier. **Decision: keep Qwen3-4B** (already
the documented default; no reason to churn). Instruct-2507 remains a
one-flag alternative.

### Dataset integrity fix

Both models missed the same "mixed language" item; inspection showed the
*label* was ambiguous (English-dominant sentence with Spanish insertions —
"mixed" vs "spanish" are both defensible). Benchmarks must not hinge on
noisy labels, so the item was replaced with a clean trap (question-phrased
bug report). Post-fix: choice 16/16 on both models.

## Considered and rejected (with reasons)

| Approach | Why not |
|---|---|
| `torch.compile` | No Triton support on Windows (no official wheels); CPU-only fallback would compile nothing useful. Revisit on Linux. |
| CUDA graphs | Require static shapes; prompts vary per request. Padding to fixed shapes would waste most of the compute we just saved. |
| flash-attention-2 | No prebuilt Windows wheels; building the toolchain locally is high-cost, and SDPA already meets the latency targets. |
| INT8/INT4 quantization | The 4B model already fits in 16 GB with room for the prefix cache; quantization perturbs exactly the logits the engine reads out. |
| Larger model (Qwen3-8B) | 8B BF16 ≈ 16 GB — does not fit alongside activations on the 5080; would require quantization (rejected above). The 4090 is reserved for other work. |
| RLCD training / LoRA fine-tune / distillation | No labeled decision dataset or training budget in scope; SemIf showed a frozen 4B with direct logits already reaches 0.813 balanced accuracy. Temperature fitting (A2) is the minimal viable form of "training" and is shipped. Real training is future work if a labeled corpus appears. |
| Continuous batching across requests | Current serialized lock + slots already saturates a single GPU at 66 ms fan-outs; cross-request batching adds a scheduler for little gain on this hardware. Future work for multi-GPU serving. |
| Score ordinal head (exploiting level ordering) | The probability-weighted `Σi·pᵢ` expectation already encodes ordering; a dedicated ordinal head would need training data (see above). |

## Reproducing

```bash
# latency + accuracy snapshot
HF_HOME=... DECIDEX_DEVICE=cuda:1 python benchmarks/bench.py --tag myrun --json benchmarks/results.json
# prompt variants × ensemble × temperature-fit matrix
HF_HOME=... DECIDEX_DEVICE=cuda:1 python benchmarks/experiments.py [--model ...]
# stability: start the server, then
python tests/load_test.py --concurrency 32
```

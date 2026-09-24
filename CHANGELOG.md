# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Post-v1.0.0 full-project review (code-review / ponytail-review /
architecture pass).

### Fixed (service)

- **Input-token limit now gates before GPU work**: usage is estimated and
  checked before the engine forward pass, so oversized requests are
  rejected with 422 without a wasted evaluation. `DECIDEX_MAX_INPUT_TOKENS=0`
  now means "unlimited" instead of crashing the request.
- **`round_distribution` rewritten as largest-remainder rounding** with
  argmax priority: probabilities can never go negative (possible before at
  80+ options when most entries round up), always sum to exactly 1, and the
  reported winner never flips.
- Malformed `Content-Length` headers are rejected as oversized (413)
  instead of raising a 500; API-key comparison is now constant-time
  (`hmac.compare_digest`).
- Choice usage accounting now includes option keys (they are part of the
  scored prompt text). `build_engine("stub", model=...)` no longer silently
  ignores the model name.

### Fixed (website)

- Home "Get Started" pointed at a nonexistent `/api` route (blank page) —
  now `/usage`; unknown hash routes redirect to Home instead of rendering
  blank.
- Site assets (`logo.svg`, `architecture.svg`) used absolute `/decidex/`
  paths that 404 outside a same-named subpath — now relative, working in
  dev, root hosting, and subpath hosting.
- CodeBlock string-highlighting regex never matched any string literal —
  fixed; strings now highlight site-wide.
- Fact alignment with `COMPARISON.md`/`REPRODUCE.md`: 86-question scored
  corpus (was "87" in places), "Noul MAE 0.129 → 0.061" instead of a
  "−48%" figure that belonged to an unreleased config, generation-round
  call counts, "~50 ms" hero claim.

### Fixed (docs)

- `REPRODUCE.md` referenced a nonexistent `benchmarks/distill_dataset_v4.jsonl`
  (correct target: `/tmp/train-core.jsonl`); README "choice 16/16" in the
  84/86 corpus context corrected to 23/23; README env-var table completed
  (`DECIDEX_MODEL`, `DECIDEX_MAX_BODY_BYTES`) and its detached `HF_HOME`
  row re-attached.

### Removed (website)

- Unused dependencies (six `@fontsource/*` packages, `framer-motion`),
  unimported `App.css`, dead duplicate `.code-block` CSS block and never-
  emitted syntax classes, dead fail-branch in the Compatibility SDK list,
  scaffold `my-app` package name.

## [3.0.0] - 2026-09-24

**Renamed JevLike → Decidex** (the old name collided with the unrelated
`vinnylarouge/jevlike` project). New identity: package `decidex`, model ids
`decidex-latest`/`decidex-1.0.0`, env vars `DECIDEX_*`, CLI `decidex`,
GGUF artifacts `decidex-*.gguf`. Official Jev aliases (`jev-latest`) stay
accepted.

### Added (release infrastructure)

- **Six-language docs**: full README in English / 简体中文 / 日本語 and core
  READMEs in Español / Français / Deutsch.
- **Reproducibility pack**: `REPRODUCE.md` (env → data → training → eval →
  GGUF → publishing, with the exact command lines of the released lineage)
  and all four distillation datasets committed in-repo (26 MB, 17,954
  samples).
- **HuggingFace release pipeline**: model cards under `hf-release/`
  (semantic adapter names — **decidex-core-8b** = balanced flagship,
  **decidex-true-8b** = noul-perfect, **decidex-draft-0.6b**, GGUF builds
  with `r1` lineage markers, distillation dataset),
  token-based `scripts/upload_hf.py`, `scripts/fetch_models.py` for
  consumers. Published under huggingface.co/fly88oj.
- **Signed git history**: dedicated ed25519 GPG signing key; commits and
  tags signed.
- Dedicated ed25519 SSH key for HF uploads (`~/.ssh/id_ed25519_decidex_hf`,
  wired into `~/.ssh/config` for `hf.co`).

## [2.4.0] - 2026-09-23

### Added

- **Replications survey** (`REPLICATIONS.md`): landscape of all known Jev
  replication projects (12, across five technical routes — frozen-logit
  readout, trained decision heads, from-scratch scorers, parallel
  constrained decoding, prompt wrappers), sourced from GitHub/HF/Reddit/
  apidog/Classmethod roundups plus first-hand pages. Per-project comparison
  (base model, training, readout, primitives, API compatibility, measured
  official agreement) and Decidex's position: the only project distilling
  from the official API's actual outputs, the only contract-level official
  compatibility (openapi + both official SDKs verified), and the only
  GGUF+MTP export; Open-Jev (ZefanCai) leads the trained-decision-head
  route and SemIf the frozen-readout route.

## [2.3.0] - 2026-09-23

### Added

- **GGUF usage guide + decision client**: `GGUF-USAGE.md` covers
  llama-server / Ollama / LM Studio (raw completion prompt, greedy decoding,
  letter readout; logprobs for full probability distributions).
  `examples/gguf_decision_client.py` implements the three Jev primitives on
  top of llama-server's `n_probs` and is live-verified: the Q4 build through
  a CPU llama-server returns noul 0.9503 / score 1.0404 on the official
  docs example — matching the official answers (0.95 / 1.05).

## [2.2.0] - 2026-09-23

### Added

- **MTP / speculative-decoding support for the GGUF builds**: a distilled
  draft model (`gguf/decidex-draft-0.6b-Q8_0.gguf`, Qwen3-0.6B + LoRA
  trained on the same 16k decision corpus) lets all three quantized builds
  run llama.cpp draft-based MTP via `--model-draft ... --spec-draft-n-max`.
  Qwen3-8B has no checkpoint NextN weights, so the draft-model route is the
  MTP mechanism (same one llama.cpp uses for MTP heads internally). All
  three target+draft combinations load- and generate-verified with
  llama-cli b11124.

## [2.1.0] - 2026-09-23

### Added

- **GGUF builds of the v7 adapter** (merged into Qwen3-8B): Q4_K_M (5.0GB),
  Q6_K (6.7GB), Q8_0 (8.7GB) + F16 master (15.6GB) under `gguf/`, produced
  via llama.cpp (`convert_hf_to_gguf.py` + `llama-quantize`). Verified
  structurally (arch qwen3, 399 tensors, expected quant-type mixes per
  level). Usage notes and regeneration commands in the README.

## [2.0.0] - 2026-09-23

### Added

- **Distillation data de-skewing at scale** (user-requested audit): the
  v1-v4 corpus was heavily skewed (choice 10%, only 4-5-option choices,
  3-5-level scores, 100% string states, one domain, 24 fixed noul
  templates, 4% CJK). `distill_generate_v2.py` produces balanced data
  (50/25/25 kinds, choice arities 2-26, score levels 2/3/10, 24%
  structured dict/list states with backtick-path questions, 8 domains,
  ~12% CJK); +8971 samples in one zero-failure batch. Total corpus now
  17,954 samples across four mining/generation rounds.
- **Adapter v8** (`adapters/decidex-true-8b`): trained on the balanced
  corpus — **noul decision agreement 52/52 (perfect)** on the comparison
  corpus, best noul MAE 0.0609. v7 remains the default (best total
  84/86, choice 16/16); three-adapter trade-off table in
  `COMPARISON.md` §10. Generation-distribution disagreement rate vs the
  official dropped 26% → **4.6%**.
- Active miner now accepts --model/--dtype/--lora (mines any engine).

## [1.9.0] - 2026-09-23

### Changed

- **Distillation training moved to the idle 4090** (user direction) and
  retrained at full BF16 precision with LoRA r=32 (`adapters/decidex-core-8b`):
  noul decision agreement **0.9808** (51/52 vs the official answers — the
  best configuration measured by a wide margin), choice 16/16, score modal
  agreement 0.909, 75ms latency. The adapter also transfers to the int4
  base for 16GB GPUs (0.9231), so one adapter serves both cards.

## [1.8.0] - 2026-09-23

### Added

- **Quantized 8B tier**: engine + CLI accept `--dtype int8|int4`
  (bitsandbytes); Qwen3-8B fits the 16GB GPU and, with the QLoRA-distilled
  adapter (`benchmarks/lora_adapter_v6_8b`), becomes the best-agreement
  configuration measured: noul probability MAE −48% vs the 4B baseline,
  decision agreement 0.9231, choice 16/16, 89ms single-question latency
  (full table in `COMPARISON.md` §9). Training gained QLoRA support
  (quantized base + gradient checkpointing + kbit prep).
- Note: Qwen3.8 (27B multimodal / 2.4T MoE) does not fit 16GB; documented
  as considered-and-rejected with the 8B tier as the practical path.

## [1.7.0] - 2026-09-23

### Removed

- Hybrid routing (`DECIDEX_HYBRID*`) — the maintainer's direction is a pure
  local replication, not an official-API router. Server code, tests, and the
  README row are gone; the live measurements remain in `COMPARISON.md` §8 as
  history.

## [1.6.0] - 2026-09-23

### Added

- **Active-learning distillation loop** (`benchmarks/distill_active.py`):
  mines fresh samples where local and official disagree (decision flips and
  probability-distance thresholds), keeps 20% agreements against forgetting.
  Round 1 (591 disagreements) produced adapter **v4** — the shipped
  recommendation: score modal agreement 0.727→**0.818**, noul MAE
  0.129→0.113, decision ties baseline. Round 2 over-mined into hedging
  (documented negative result in `COMPARISON.md` §8).
- **Hybrid routing** (`DECIDEX_HYBRID=1`): uncertain answers (low
  confidence, or noul inside the uncertainty band) forward the whole request
  to the official API; failures fall back to local answers. Verified live
  in all three paths (forced forward / fallback / natural routing). Resolves
  the residual ambiguous-judgment disagreements by construction.
- Unit tests for the hybrid routing predicate and config (7 new).

## [1.5.0] - 2026-09-22

### Added

- **Official-distillation LoRA pipeline** (user-approved): batch-generate
  distillation targets from the official API (`distill_generate.py`, 1300
  requests, <$0.05), soft-label CE training on the engine's letter-readout
  position (`distill_train.py`, LoRA r=16, 0.24% params), optional adapter
  loading in the engine/CLI (`--lora` / `DECIDEX_LORA`). Three training
  iterations; full-corpus verdict in `COMPARISON.md` §7: noul probability
  MAE −24% (0.128→0.098), decision agreement ties the base model (0.885,
  already above the 0.85 bar), small choice/score trade-offs — the adapter
  ships as an opt-in for probability-shape fidelity, base model stays default.

## [1.4.0] - 2026-09-22

### Added

- **Alignment campaign toward the official API** (`benchmarks/fit_to_official.py`):
  the official answers from the real comparison are used as fitting targets
  with a fit-half/validate-half split. Eight avenues tested — noul option-text
  variants, Platt scaling, per-type/global temperature fitting, dual-text
  ensembling, score level anchoring, and two alternative base models
  (Qwen3-4B-Instruct-2507, Qwen3.5-4B). **None transferred to the held-out
  half**; the shipped configuration is already at the maximum achievable
  consistency for a 4B-class local model (choice L1 0.0002; noul decision
  agreement 85–88.5%, at the ceiling SemIf independently measured for
  open 4B models). Evidence and per-avenue numbers in `COMPARISON.md` §6.

## [1.3.0] - 2026-09-22

### Added

- **Measured comparison against the official Jev API** (via OpenRouter's
  `/api/alpha/decisions` gateway, model `typesafe/jev-1.13-20260917`):
  `benchmarks/compare_corpus.py` (54 requests / 87 questions / 8 invalid
  probes) + `benchmarks/compare_official.py` (dual-endpoint runner with
  agreement, calibration, self-consistency, question-count invariance,
  error-contract, and token-metering metrics; `--self-test` proves the
  harness on a perfect-agreement run). Findings in `COMPARISON.md`:
  choice top-1 agreement 23/23 with near-identical distributions; noul
  median |Δp| 0.04 with root-caused outliers (bare yes/no semantics —
  resolved by explicit criteria, measured 0.82→0.05 — and the official's
  softer calibration); official metering counts ~7x the request text
  (internal scaffolding); the OpenRouter gateway normalizes native 422s
  to 400s.

## [1.2.0] - 2026-09-22

Systematic optimization campaign — every change measured on the 53-item
authored benchmark (`benchmarks/`), full log in `OPTIMIZATION.md`.

### Added

- **KV prefix reuse**: one forward per shared state; every question (and
  every relevance probe) continues from that KV cache. Long-document
  fan-outs stop OOM-ing and repeat queries hit a cross-request LRU prefix
  cache (`DECIDEX_PREFIX_CACHE_GB`, default 4 GB): 3.3k-token state × 10
  questions 2197 ms cold → 193 ms warm. Equivalence pinned by a regression
  test (argmax match, prob deviation < 0.02).
- **Benchmark suite** (`benchmarks/`): 53 authored items incl. hard
  sarcasm/negation/near-miss cases, accuracy + calibration (Brier, stated-
  confidence gap) + latency (single, fan-out, long-state fan-out) metrics,
  and an experiment matrix runner (prompt variants × ensembling ×
  temperature fitting).
- **Stability**: evaluation runs in the threadpool behind a GPU lock and a
  slot semaphore (`DECIDEX_MAX_QUEUE`, default 8); overload returns 429 with
  `Retry-After`; CUDA OOM halves the batch and retries instead of crashing.
  Load-tested at 32 concurrent requests: zero 5xx, all succeed via the
  documented retry path, `/health` p50 2 ms under full GPU load
  (`tests/load_test.py`).
- Engine A/B knobs: `prompt_variant` (plain/strict/chat) and
  `ensemble_rounds` (option-order self-consistency; hard-rubric score
  accuracy 0.846 → 1.00 at 3× latency on those jobs).

### Changed

- Letter-readout batch chunk raised 8 → 16 (measured 20% faster fan-outs,
  flat beyond); OOM backoff bounds memory when 16 does not fit.
- 10-question fan-out latency 150.6 → 65.9 ms; single question 52.8 →
  49.1 ms (RTX 5080, Qwen3-4B).
- `torch.inference_mode()` throughout the engine.

### Benchmark findings (documented in `OPTIMIZATION.md`)

- Chat-template prompts are catastrophic for direct logit readout on
  thinking-hybrid models (noul drops to coin-flip); completion-style wins.
- Temperature fitting (single-parameter calibration) is unstable on small
  labeled sets — τ=1.0 ships as default, the fitting tool is included.
- Qwen3-4B-Instruct-2507 matches Qwen3-4B exactly on this benchmark;
  the default is unchanged.

## [1.1.0] - 2026-09-22

### Added

- **Verified drop-in compatibility with both official SDKs.** `typesafe-sdk`
  (Python, 0.7.0) and `@typesafe-ai/sdk` (JS, 0.6.0) run against a Decidex
  server by changing only the base URL (or, for the Python SDK, purely via
  `TYPESAFE_BASE_URL`/`TYPESAFE_API_KEY` env vars). Run
  `tests/check_official_sdk_compat.py` against a live server to re-verify;
  the official SDK is available via the `compat` extra.
- Open-source scaffolding: `LICENSE` (MIT), `CONTRIBUTING.md`, this
  changelog, `.gitignore`, ruff lint configuration (`F,E,W,I`, line length
  110) with a clean baseline.
- Trilingual README (English base, `README.zh-CN.md`, `README.ja.md`) and a
  bilingual research document (`RESEARCH.md` in English, `RESEARCH.zh-CN.md`
  in Chinese).

### Fixed

- **High-cardinality result alignment**: when one request mixed a >26-option
  Choice with other questions, the LLM engine appended direct-readout results
  before relevance-probe results, misaligning answers with question ids.
  Results are now written by job index; regression test included.
- Left-padded batch forward passes now pass explicit `position_ids` derived
  from the attention mask, keeping non-RoPE models correct.
- `/health` is exempt from Bearer auth so monitoring probes keep working when
  an API key is configured.
- SDK retries connection-level failures (transport errors and timeouts), not
  just HTTP 429/529.

### Changed

- **Contract aligned with the official `openapi.json`** (embedded in
  `typesafe-sdk` as generated schemas): `GET /v1/models` now returns the
  official `ModelMetadataList` shape (`{models: [{name, description,
  release_date}]}`); 422 bodies use the official FastAPI style
  (`{"detail": [{"loc", "msg", "type"}]}`), with 401/529 using `{"detail": "..."}`;
  `instructions` is optional on all question types; noul `criteria.true/false`
  accept null; score `legend` echoes criteria entries verbatim (structured
  values included); Score accepts 1–26 levels (openapi floor is 1; the cap of
  26 is the letter-readout capacity, returned as a clear 422 beyond that).
- Model id bumped to `decidex-1.1.0`; `decidex-1.0.0` remains accepted.
- Relevance probes for >26-option questions read the same option-letter
  logits as the direct path (previously "yes"/"no" token logits) — one
  readout mechanism, sharper signal.
- `demo` subcommand defaults to the dependency-free `stub` engine so the
  quickstart works before any ML extras are installed.
- `make_stub_engine` factory replaced by the `StubEngine` class.
- Base `Engine.token_count` returns a chars/4 estimate by default; engines
  with a tokenizer override it.

## [1.0.0] - 2026-09-21

### Added

- `POST /v1/systemone` HTTP API replicating the official TypeSafe System One
  contract: `state` + typed `questions` in, `{model, answers, usage}` out,
  with 401/422 error semantics and field-level validation messages.
- Three question primitives with official semantics:
  - **Choice** — up to 255 options, returns `choice` / `probabilities` / `confidence`.
  - **Score** — 2–10 ordered levels, returns probability-weighted `score`
    (can land between levels), `legend`, `probabilities`, `confidence`.
  - **Noul** — yes/no probability in `[0, 1]`.
- Official confidence statistic `clamp((K·p_max − 1)/(K − 1), 0, 1)`, taken
  from the interactive demo source on the official docs and verified against
  both worked examples in the official API reference.
- Parallel, independent evaluation of all questions in one request.
- `DecidexClient` SDK with `Choice` / `Score` / `Noul` builders,
  interface-compatible with the official `typesafe_sdk` (change the import
  and base URL only); retries 429/529 with exponential backoff and
  `Retry-After` support.
- Pluggable engines:
  - `llm` — direct typed-logit readout from a frozen causal LM (default
    Qwen3-4B): options labeled A/B/C…, one forward pass, zero generated
    tokens; >26 options switch to per-option relevance probes. This is the
    approach validated by the community replication SemIf (formerly OpenJev).
  - `embedding` — sentence-transformer similarity fallback (CPU-friendly).
  - `stub` — deterministic lexical engine for tests and dependency-free runs.
- `python -m decidex serve|demo` CLI with engine/model/device/temperature/API-key options.
- Model aliases: `decidex-latest` / `decidex-1.0.0` and the official
  `jev-latest` / `jev-1.13.0`, so unmodified official-SDK code can point at
  a Decidex server by changing only the base URL.
- `GET /v1/models`, `GET /health` endpoints; optional Bearer auth.
- 63 tests: 58 fast (contract / math / SDK / stub, no ML dependencies) +
  5 slow (live engine smoke tests, including the 30-option relevance path).
- Bilingual research document (English / Chinese) recording the official and
  community evidence behind every design decision.
- Trilingual README (English base, Chinese, Japanese).

# Decidex

**State in. Typed decisions out. One forward pass.**

A local, open reimplementation of the Jev (TypeSafe System One) decision
model & API — no text generation, no parsing, calibrated probabilities with
every answer, trained toward the official model's actual outputs.

**English** | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

[Research notes](RESEARCH.md) · Official API reference: docs.typesafe.ai/api · [Replications survey](REPLICATIONS.md) · [Reproduce everything](REPRODUCE.md)

**Decidex** is an open, local reimplementation of [TypeSafe AI's "Jev"](https://typesafe.ai/blog/introducing-system-one-models-and-jev):
**unstructured state in, typed calibrated decisions out.** No text
generation, no JSON repair, no hallucinated formats — questions and answers
are typed values you define up front, every question is evaluated
**independently and in parallel** against the same state, and one call
returns in milliseconds.

```
┌─────────────┐   state (string / JSON)     ┌──────────────────┐
│ your code   │ ─────────────────────────► │  Decidex service │
│ (if/routing)│   questions (Choice/Score/  │  ┌────────────┐  │
└─────────────┘   Noul, any number)         │  │ frozen LM  │  │ one forward
       ▲                                    │  │ direct     │  │ pass reads
       │   answers: typed values + probs    │  │ logits     │  │ probabilities
       └────────────────────────────────────└──└────────────┘──┘
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[all]"

# Dependency-free smoke test (lexical stub engine)
.venv/Scripts/python -m decidex demo --engine stub

# Start the service (default: Qwen3-4B direct-logit readout; downloads the
# model on first run; point HF_HOME at a large drive)
HF_HOME=/path/to/hf-cache .venv/Scripts/python -m decidex serve --engine llm --port 8600
```

Call it — field-for-field identical to the official API:

```bash
curl -s http://127.0.0.1:8600/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "decidex-latest",
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds",
                                 "technical": "Bugs, outages, integrations",
                                 "sales": "Pricing, upgrades, new accounts"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

Python SDK — interface-compatible with the official `typesafe_sdk`; change
the import and base URL and existing code runs against your local server:

```python
from decidex import Choice, DecidexClient, Noul, Score

with DecidexClient() as client:          # http://127.0.0.1:8600
    response = client.system_one(
        state={"message": "I was charged twice for order A-104.", "order": {"id": "A-104"}},
        questions={
            "refund_requested": Noul("Does `message` request a refund?"),
            "request_type": Choice("What is the main request in `message`?",
                                   criteria={"refund": "Wants money returned",
                                             "information": "Asking a question",
                                             "other": "Anything else"}),
            "frustration": Score("How frustrated is the customer?",
                                 criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )
    print(response.answers["refund_requested"].noul)      # 0..1
    print(response.answers["request_type"].choice)        # "refund"
    print(response.answers["request_type"].confidence)    # 0..1
    print(response.answers["frustration"].score)          # 0..2, may land between levels
```

Full worked example (ticket triage with confidence-gated routing, the
official intent-routing pattern): [`examples/ticket_triage.py`](examples/ticket_triage.py)

## The three primitives

| Type | criteria | Returns | Semantics |
|---|---|---|---|
| `choice` | `map<option, description|null>`, ≤255 entries | `choice` + `probabilities` + `confidence` | selected option = argmax of probabilities |
| `score` | 1–26 ordered level descriptions (official prose recommends 2–10) | `score` + `legend` + `probabilities` + `confidence` | `score = Σ i·p_i` (may land between levels) |
| `noul` | optional `{true, false}` descriptions | `noul` ∈ [0,1] | probability the answer is yes; the value itself is the belief, no confidence |

`confidence = clamp((K·p_max − 1)/(K − 1), 0, 1)` — the formula published in
the interactive demo source of the official docs, verified against both
worked examples in the official API reference (see
[RESEARCH.md](RESEARCH.md#12-the-three-primitives-the-whole-api)).

## Compatibility with the official API

**Verified drop-in.** Both official SDKs run against a Decidex server by
changing only the base URL — no other code changes:

- **Python** `typesafe-sdk` (tested with 0.7.0): `TypeSafeClient(api_key="x", base_url="http://127.0.0.1:8600")`, or zero code change at all via `TYPESAFE_BASE_URL` + `TYPESAFE_API_KEY` env vars. The official docs' examples work verbatim, including the grouped `.nouls` / `.choices` / `.scores` accessors, dict-style questions, null choice descriptions, and `client.models.list()`.
- **JavaScript/TypeScript** `@typesafe-ai/sdk` (tested with 0.6.0): `new TypeSafeClient({ apiKey: "x", baseURL: "http://127.0.0.1:8600" })` — `systemOne` / `models.list` / typed answers all pass.
- Raw HTTP matches the official shapes too, including the FastAPI-style
  `{"detail": [{"loc", "msg", "type"}]}` 422 body (the official contract is
  embedded in `typesafe-sdk` as generated OpenAPI schemas; Decidex follows it).

Contract details aligned with the official `openapi.json`:

- `POST /v1/systemone`, `GET /v1/models` — same paths, same request/response fields (`model` / `answers` / `usage`).
- `GET /v1/models` returns the official `ModelMetadataList` shape: `{models: [{name, description, release_date}]}`.
- Error semantics: `401`, `422` (FastAPI-style detail naming the offending field), `429`/`529` (retryable; the SDK backs off exponentially and honors `Retry-After`).
- `instructions` is optional on all three question types; noul `criteria.true/false` accept null; score `legend` echoes your criteria entries verbatim (structured values included).
- Model names: `decidex-latest`/`decidex-1.1.0` plus the official `jev-latest`/`jev-1.13.0` aliases.
- Validation limits: Choice ≤255 options; Score accepts 1–26 levels (official openapi allows ≥1, docs prose recommends 2–10 — we enforce the letter-readout cap of 26 with a clear error).

Run the compatibility check yourself (needs a running server and
`pip install -e ".[compat]"`):

```bash
.venv/Scripts/python tests/check_official_sdk_compat.py
```

## Architecture

```
decidex/
├── server.py            FastAPI: /v1/systemone validation + parallel evaluation + official response shapes
├── sdk.py               DecidexClient + Choice/Score/Noul (mirrors typesafe_sdk)
├── calib.py             Official formulas: softmax, confidence, Σi·p_i, distributions summing to 1
├── render.py            state/instructions/criteria → text (string | object | array)
├── cli.py               python -m decidex serve | demo
└── engines/
    ├── llm_logits.py    ★ default engine: one forward pass reads option-letter logits from a frozen LM (the SemIf approach)
    ├── embedding.py     alternative: sentence-transformer cosine similarity (CPU-friendly, heuristic)
    └── base.py          Engine interface + dependency-free lexical stub (for tests)
```

**The LLM direct-logit engine** (the core reproduction): options are
labeled `A./B./C.…` in the prompt; a single forward pass reads the logits of
the option letters at the final position and softmaxes them into
probabilities — **no tokens are sampled and nothing is generated**, which is
the mechanical equivalent of Jev's "parallel sampler, free outputs" (the
community replication SemIf measured the same scheme at 0 output tokens and
5.2× faster than autoregressive JSON). All questions in a request share the
state and are scored in padded parallel batches; beyond 26 options the
engine switches to per-option relevance probes with renormalization — the
same two-stage shape TypeSafe describes for high-cardinality choices.

## Measured on the development machine (RTX 5080 / Qwen3-4B)

| Scenario | Before optimization | After optimization |
|---|---|---|
| Single question (warm p50) | 53 ms | **49 ms** |
| 10-question parallel fan-out (warm) | 151 ms | **66 ms** |
| 10 questions over a 3.3k-token document, cold | OOM crash | **2.2 s** |
| …same document again (prefix cache hit) | — | **193 ms** |
| 32 concurrent requests | — | all succeed, `/health` p50 2 ms under load, zero 5xx |

**Latency context** — these are local-GPU numbers (no network). The
official Jev API's published range is 70–500 ms (TypeSafe-reported, US West
Coast direct); independent measurements put it at ~300 ms p50 for direct
API calls (Open-Jev project) and ~380 ms p50 via OpenRouter (Reddit
r/LocalLLM benchmark). Local GPU vs remote API are not directly
comparable — Decidex's advantage is privacy, offline use, and zero
per-call cost, not raw speed over the network.

Accuracy on the 53-item authored benchmark (including hard sarcasm /
negation / near-miss items): noul 24/24, choice 16/16 (calibration gap
−0.0009), score 84.6% hard items — 100% with the opt-in
`ensemble_rounds=3`. Every optimization and rejection is documented with
measurements in [OPTIMIZATION.md](OPTIMIZATION.md).

A measured head-to-head against the official API (via OpenRouter):
[COMPARISON.md](COMPARISON.md) — choice top-1 agreement 23/23, distribution
JS divergence 0.0025; documented root causes for the remaining noul/score gaps.

## Configuration

| Env var / flag | Default | Notes |
|---|---|---|
| `--engine` | `llm` (serve) / `stub` (demo) | `llm` \| `embedding` \| `stub` |
| `--model` / `DECIDEX_MODEL` | engine default (Qwen/Qwen3-4B / all-MiniLM-L6-v2) | HF model name or local path; any causal LM works |
| `--temperature` | 1.0 (llm) / 0.05 (embedding) | probability sharpness; raise to soften overconfident logits |
| `--device` / `DECIDEX_DEVICE` | auto cuda→cpu | pick a GPU on multi-GPU machines, e.g. `cuda:1` |
| `--api-key` / `DECIDEX_API_KEY` | none (open locally) | when set, Bearer auth is enforced |
| `DECIDEX_BASE_URL` | `http://127.0.0.1:8600` | SDK default target |
| `DECIDEX_MAX_INPUT_TOKENS` | 65536 | server-side context cap (the llm engine additionally enforces its own 8192; the smaller wins) |
| `DECIDEX_PREFIX_REUSE` | 1 | KV prefix reuse: one forward per shared state instead of one per question |
| `DECIDEX_PREFIX_CACHE_GB` | 4 | LRU budget (GB) for cross-request state-prefix KV caches; 0 disables |
| `DECIDEX_MAX_QUEUE` | 8 | max concurrently-waiting evaluations; beyond that, 429 + `Retry-After` |
| `DECIDEX_MAX_BODY_BYTES` | 10485760 | request-body byte cap (DoS floor, rejected with 413) |
| `--lora` / `DECIDEX_LORA` | none | optional LoRA adapter distilled from the official API (see COMPARISON.md §7) |
| `--dtype` | `auto` | `int8` / `int4` bitsandbytes quantization — fits 7–8B models in 16GB |
| `HF_HOME` | `~/.cache/huggingface` | model cache; point at a large drive |

**Recommended tiers** (measured against the official API, `COMPARISON.md` §9-10):
latency `Qwen3-4B` (~49ms, decision agreement 0.885); 16GB GPUs
`Qwen3-8B --dtype int4 --lora benchmarks/adapters/decidex-core-8b` (~80ms,
0.923); best agreement on a 24GB GPU `Qwen3-8B --lora
benchmarks/adapters/decidex-core-8b` (~64ms, total 84/86, choice 23/23,
score modal 0.909) or `adapters/decidex-true-8b` (noul 52/52 perfect,
noul MAE 0.061).
> Network tip: if PyPI is slow from your network, install with
> `-i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir`; HuggingFace
> direct downloads generally work.

## Differences from the official Jev (honest list)

| Aspect | Official Jev | Decidex |
|---|---|---|
| Model | undisclosed in-house architecture | frozen open LM (default Qwen3-4B; `--model` accepts any causal LM) |
| Training | RLCD (RL for calibrated decisions) | none; probabilities come from logits + a temperature knob |
| Calibration | officially calibrated per-answer confidence | single temperature knob; formulas match but calibration quality depends on the base model |
| Latency | 70–500 ms (dedicated service) | 53 ms single noul / 130 ms 10-question fan-out (measured, 5080 + 4B) |
| Context | 64k tokens | 8k by default for the llm engine (`max_input_tokens`), 64k server cap |
| Token accounting | exact | exact when the engine has a tokenizer, else chars/4 estimate |

## Tests

```bash
.venv/Scripts/python -m pytest tests -m "not slow"   # contract+math+SDK, no ML deps, seconds
.venv/Scripts/python -m pytest tests -m slow          # live engine smoke tests (needs the ML stack)
```

## GGUF builds (llama.cpp / Ollama / LM Studio)

The recommended adapter (v7, merged into Qwen3-8B) ships as GGUF in `gguf/`
(gitignored — regenerate with the commands below):

| File | Size | Best for |
|---|---|---|
| `decidex-core-8b-r1-Q4_K_M.gguf` | 5.0 GB | 16GB GPUs / 16GB+ RAM CPU |
| `decidex-core-8b-r1-Q6_K.gguf` | 6.7 GB | quality-first |
| `decidex-core-8b-r1-Q8_0.gguf` | 8.7 GB | closest to BF16 behavior |
| `decidex-core-8b-r1-f16.gguf` | 15.6 GB | master for re-quantizing |

Full per-platform usage guide (llama-server with probability readout, Ollama,
LM Studio): [GGUF-USAGE.md](GGUF-USAGE.md). Ready-made client:
`examples/gguf_decision_client.py` (verified live: noul 0.9503 / score 1.0404
vs official docs' 0.95 / 1.05).

These are decision-engine models: use them with the same completion prompt
Decidex's engine renders (state + question + lettered options + `Answer:`),
greedy decoding, and read the first letter. Serving through Decidex itself
still uses the HF/PEFT path (`--model <merged> --lora ...`) — the GGUFs are
for the llama.cpp ecosystem (llama-server/Ollama/LM Studio).

Regenerate: merge adapter (`PeftModel.merge_and_unload`) →
`llama.cpp/convert_hf_to_gguf.py <merged> --outtype f16` →
`llama-quantize <f16> <out> Q4_K_M|Q6_K|Q8_0`.

### MTP (speculative decoding)

Qwen3-8B checkpoints ship no NextN/MTP weights (only Qwen3-Next / Qwen3.5 /
Qwen3.8-class hybrids do), so MTP for these GGUFs uses llama.cpp's
draft-model speculative decoding — the same mechanism an MTP head uses
internally. The repo ships a **distilled draft**: Qwen3-0.6B trained on the
same decision corpus (letter distribution matched to the target, so draft
acceptance is high):

| File | Size |
|---|---|
| `gguf/decidex-draft-0.6b-Q8_0.gguf` | 0.6 GB |

Launch any of the three quants with MTP (verified: all three load and
generate with the draft attached):

```bash
llama-server -m gguf/decidex-core-8b-r1-Q4_K_M.gguf   --model-draft gguf/decidex-draft-0.6b-Q8_0.gguf   --spec-draft-n-max 4 -ngl 99          # GPU; tune 2-8
# same for Q6_K / Q8_0 targets
```

Honest notes: speculative decoding accelerates **generation**; Decidex's
native logit-readout mode generates zero tokens, so MTP benefits
llama-server chat/completions use of the merged model, not the decision
API itself. Benchmark speedups on your GPU — on CPU the draft adds
overhead per step.

## Acknowledgments & license

- Built from public evidence: TypeSafe AI's docs and blog (API shapes,
  formulas) and the community replication
  [SemIf](https://github.com/TheoLeeCJ/SemIf) (the direct-logit approach).
  This project is not affiliated with TypeSafe AI; Jev and TypeSafe are
  trademarks of their respective owners.
- Project code: MIT (see [LICENSE](LICENSE)). Contributing:
  [CONTRIBUTING.md](CONTRIBUTING.md). Changes: [CHANGELOG.md](CHANGELOG.md).

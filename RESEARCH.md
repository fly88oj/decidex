# Research: the implementation basis of Jev

> Researched 2026-09-21. Goal: establish what TypeSafe AI's **Jev** (System One
> model) actually is — from the official introduction and from community
> replications — and record the evidence that drives every design decision in
> this repository's `Decidex` reimplementation.

## 1. What Jev is (official evidence)

Sources: [TypeSafe AI blog, "Introducing System One Models & Jev" (2026-09-15)](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[official docs](https://docs.typesafe.ai/).

- **Released**: 2026-09-15 by TypeSafe AI (founder Diogo Almeida, co-author of
  InstructGPT/RLHF at OpenAI), early access.
- **Positioning**: the first "System One" model — the name comes from
  Kahneman's *Thinking, Fast and Slow* (fast, intuitive thinking). "Jev"
  honors William Stanley Jevons (the Jevons paradox: efficiency gains
  increase demand).
- **One line**: *"frontier-intelligence function call: unstructured state in,
  typed probabilistic decisions out."*
- **It is not a chat model**: it gives up string generation entirely, so it
  **cannot produce type errors or hallucinated formats by construction**
  (it can still answer wrongly — "the wrong valid value").

### 1.1 Mechanism vs. LLMs (from the official comparison table)

| Dimension | Traditional LLMs | Jev (System One) |
|---|---|---|
| Training | RLHF / RLVR | **RLCD** (Reinforcement Learning for Calibrated Decisions) |
| Input | unstructured text, sequential messages | unstructured data, **structured program state** |
| Output | strings (parse + validate before use) | **type-safe structured values + calibrated probabilities + confidence** |
| Sampling | autoregressive, token by token | **parallel**: one pass answers everything; output side costs nothing |
| Speed | 3–329 s | 70–500 ms |
| Price | input $0.20–10/MTok, output ≈5× input | input $0.042/MTok, output free |
| Confidence | overconfident, inconsistent | every answer carries confidence; calibrated (higher confidence ⇒ higher accuracy) |

### 1.2 The three primitives (the whole API)

From the official docs ([Primitives](https://docs.typesafe.ai/primitives),
[API reference](https://docs.typesafe.ai/api)):

| Question type | Asks | criteria | Returns |
|---|---|---|---|
| **Choice** | pick one from a set | `map<option, description|null>`, **≤255 options** | `choice` (argmax), `probabilities` (sum to 1), `confidence` |
| **Score** | rate on a rubric | ordered array, **2–10 level descriptions** | `score` (probability-weighted position, may land between levels, e.g. `1.05`), `legend`, `probabilities`, `confidence` |
| **Noul** | is this statement true? | optional `{true, false}` descriptions | `noul` ∈ [0,1] (probability of yes; no confidence — the value *is* the belief) |

Semantics confirmed verbatim from the official docs:

- `score` is the **probability-weighted level index**: `score = Σ p_i · i`.
  The official example `{0: 0.0, 1: 0.95, 2: 0.05} → score 1.05` checks out:
  `0×0 + 1×0.95 + 2×0.05 = 1.05` ✓
- `confidence` is **derived** from the probability distribution. The official
  docs' interactive demo ships its exact formula (from the page source):
  ```js
  confidence = clamp((K · p_max − 1) / (K − 1), 0, 1)   // K = number of options/levels
  ```
  i.e. "certainty in excess of the uniform distribution". Verified against
  both worked examples in the official API reference: K=3, p_max=0.88 → 0.82
  (docs show 0.81); p_max=0.95 → 0.925 (docs show 0.92) ✓
- All questions in one call are evaluated **in parallel and in isolation**
  against the same `state`; adding questions barely changes response time
  (the "speculative fan-out" pattern).
- `instructions` may be a string / object / array; the object form puts the
  question in one field and data in the others, referencing `state` with
  backtick paths (e.g. `` `ticket.messages[0].text` ``).
- `state` may be a string, JSON object, or array of text. Text only — no
  images, audio, or video.
- Context limits: 64k tokens for state + all questions together; 32k for
  state + the single longest question.

### 1.3 The exact HTTP protocol

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

Request body (verbatim from the official reference):

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"} },
    "department": { "type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"} },
    "frustration": { "type": "score", "instructions": "How frustrated is the customer?",
                     "criteria": ["Calm", "Frustrated", "Very angry"] }
  }
}
```

Response body (verbatim):

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent": { "type": "noul", "noul": 0.95 },
    "department": { "type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                    "confidence": 0.81 },
    "frustration": { "type": "score", "score": 1.05,
                     "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                     "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                     "confidence": 0.92 }
  },
  "usage": {"input_tokens": 304, "output_tokens": 18}
}
```

Errors: `401` (invalid key), `422` (body validation, response names the
offending field), `429` (rate limits: 250k tokens/s, 1200 req/min), `529`
(overloaded; retry with backoff). The alias `jev-latest` currently resolves
to `jev-1.13.0`; the response's `model` field reports the versioned id that
answered.

SDKs: Python `typesafe-sdk` (`TypeSafeClient().system_one(state=...,
questions={...})` with `Choice`/`Score`/`Noul` builders) and JS
`@typesafe-ai/sdk` (`choice()/score()/noul()` functions). Answers come back
under the question ids as typed objects.

### 1.4 The machine-readable contract: openapi.json via the official SDK

The official `typesafe-sdk` package ships the official API's `openapi.json`
as generated Pydantic schemas (`typesafe_sdk._schemas.models`, generated by
datamodel-codegen). This is stricter evidence than the docs prose and
corrected several assumptions when Decidex was aligned for drop-in use
(2026-09-22):

- `GET /v1/models` returns `ModelMetadataList`: `{models: [{name,
  description, release_date}]}` — not an OpenAI-style id/object list.
- 422 bodies are FastAPI-native: `{"detail": [{"loc", "msg", "type"}]}`;
  other errors use `{"detail": "..."}`.
- `instructions` is **optional** (`str | object | list | null`) on all three
  question types, despite the docs saying required.
- `NoulCriteria.true/false` each accept null.
- Score `criteria` has `min_length=1` and no OpenAPI maximum (the docs prose
  "2 to 10 levels" is a recommendation, not the enforced bound). Decidex
  accepts 1–26 (26 = letter-readout capacity) with a clear 422 beyond.
- Score answers' `legend` echoes the criteria entries verbatim (structured
  values included), per the ScoreAnswer schema.
- The SDK's response wrapper additionally coerces legend/probabilities keys
  to int and exposes grouped `.nouls`/`.choices`/`.scores` accessors.

### 1.5 Officially admitted weaknesses (the "jaggedness" page)

It reads literally (negations and scoping land at face value); **it cannot
count** (iterate one Noul per item instead); **dates are text, not ordered
quantities** (enumerate with a Choice, compute in code); context rot is real
(accuracy falls as irrelevant material accumulates — retrieve and filter
first); state is not treated as adversarial (prompt injection is your
problem); it does not generate anything (get candidate values via regex or a
generative model, let Jev pick). The meta-rule: *"Avoid asking the model
something code can compute exactly. Avoid hiding several judgments inside
one question."*

## 2. Community replications

### 2.1 SemIf (formerly OpenJev, by TheoLeeCJ) — the key replication

[github.com/TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) (2.6k★,
MIT). Self-described: reproduces the **interface pattern**, not Jev's
undisclosed model or training.

Core method — **direct typed-logit readout**:

```
state (unstructured) ─┐
criteria (runtime-defined) ─┼→ frozen 4B open model ──native option logits──→ probabilities
typed options ─┘        (one forward pass; no answer token is sampled)
```

- Model: Qwen3.5-4B (revision pinned; alternates MiniCPM5-2B, Qwen3-0.6B).
- Options are letter-labeled (A/B/C…); one forward pass reads the **logits
  of the option-identifying tokens at the final position**, softmaxed into
  probabilities. No JSON, no decoding loop, no repair.
- **Shared-state optimization**: one long state prefetched once (KV prefix
  reuse), then many criteria branched in parallel. Measured on a 37-state ×
  21-criterion workload: fresh scoring 2.33 decisions/s → serial prefix
  reuse 10.75 → parallel suffixes **20.03**.
- Speed (same 4B model, 21 binary criteria, RTX 3090): direct logits
  **1.023 s / 0 output tokens** vs. autoregressive JSON array 5.332 s / 111
  tokens (5.21×).
- Quality (balanced accuracy): Qwen3-0.6B 0.440, MiniCPM5-2B 0.686,
  **Qwen3.5-4B 0.813**; agreement with a TypeSafe public-eval subset 0.845
  vs. hosted Jev's 0.883. The native-reranker route (Qwen3-Reranker) was
  worse (0.625) — direct logits is the stronger general-purpose baseline.

### 2.2 Other ecosystem projects (pattern references)

All appeared within a week of launch. The shared pattern: **keep the loop,
the safety, and the arithmetic in ordinary code; use the model only for the
narrow judgment in the middle.**

- `browser-use/jev-ultrafast` (641★): browser agent. Pages become a numbered
  element table; one request picks both the operation (CLICK/TYPE/…) and its
  target; a small LLM runs only for free text. Zurich→London flight booked
  in 7.1 s / $0.0039.
- `awlevin/typesafe-computer-use`: OCR turns the screen into symbols, the
  model picks actions. $0.0002/step vs. Opus 5's $0.032. Key lesson: *"Every
  piece of reasoning the frontier model does for free has to be rebuilt here
  as deterministic state."*
- `jarrodwatts/jev-trader`: buy/sell decision once per ~300 ms block,
  ~81 ms model latency.
- `RomanSlack/jev-drone`: 500 Hz control law (code) / 50 Hz safety reflex
  (code) / 15 Hz classical CV / ~2.5 Hz tactical judgment (model, advisory
  only).
- `fhshaik/typesafe-mario`: Mario from emulator RAM as object-centric JSON;
  `devagrawal09/jev-review`: staged code review; `AbdelStark/awesome-typesafe`:
  ecosystem index.
- `1kpapers.com`: 1,018 papers over 24 topics — DeepSeek summaries $3.99,
  classifications just $0.08 (median 256 ms).

## 3. Design decisions for Decidex

| Decision | Basis |
|---|---|
| Expose a **field-for-field identical** `POST /v1/systemone` + `GET /v1/models` | Official API reference (§1.3); official SDK code can point at Decidex by changing the base URL |
| `score = Σ p_i·i`, `confidence = clamp((K·p_max−1)/(K−1),0,1)`, probabilities sum to 1 | Verified against official examples + the official docs' demo source (§1.2) |
| Questions evaluated in parallel, independently | Official semantics (§1.2); each question is one forward pass, no shared question context |
| Default engine = **direct LLM logit readout** (Qwen3-4B-class model + letter-labeled options + temperature) | SemIf measured this route as the highest quality and closest to Jev behavior (§2.1) |
| Embedding fallback engine (sentence-transformers + softmax) | CPU-only / lowest-latency environments; SemIf's model ladder shows small models still give usable probabilities |
| Noul = two-option (yes/no) readout | Noul has no confidence and *is* the probability; same readout mechanism as Choice |
| `instructions`/`criteria` accept str\|object\|array with backtick paths | Official structured-instruction semantics (§1.2) |
| 422 naming the field, 401/429 semantics aligned | Official error table (§1.3) |
| Differences from the official service recorded honestly (RLCD training cannot be replicated; probabilities come from a frozen LM's logits + temperature) | Same stance as SemIf (§2.1) |

## 4. Source list

- Official blog: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Official docs: https://docs.typesafe.ai/ (introduction / api / primitives / confidence / llms.txt / model-jaggedness)
- Official Python/JS SDK docs: https://docs.typesafe.ai/sdk/python , https://docs.typesafe.ai/sdk/javascript
- SemIf (formerly OpenJev): https://github.com/TheoLeeCJ/SemIf
- Practical guide (dev.to, Valyu): https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e
- HN discussion: https://news.ycombinator.com/item?id=49717558
- Ecosystem index: https://github.com/AbdelStark/awesome-typesafe

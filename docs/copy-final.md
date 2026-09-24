# Decidex Website Copy — Final (B base + A grafts + reviewer fixes)

## HOME

### Hero Badge
OPEN SOURCE · RUNS ON YOUR GPU · API-COMPATIBLE

### Headline
State in. Typed decisions out.
One forward pass.

### Hero Description
An open reimplementation of the Jev decision model that runs entirely on
your machine. Your data never leaves your GPU — no API key to buy, no
rate limit to hit, no cloud to trust. Clone the repo, run one command,
and you have a decision engine that answers in ~50 milliseconds.

### Key Metrics Cards
1. **49ms** — Latency — "on your GPU — no network round-trip, no queue"
2. **100%** — Drop-in Compat — "swap one URL, both official SDKs pass all 8 checks"
3. **97.7%** — Agreement — "measured against the official API, results in the repo"
4. **5.0 GB** — Smallest Build — "fits a gaming laptop, runs offline at zero per-call cost"

### Architecture Description
Think of it as a very fast, very honest if-statement. You give it
context and a set of choices. It runs one pass through a frozen language
model (Qwen3-8B with a LoRA adapter), reads the probability of each
option, and hands you the answer with a confidence score. No prose to
parse, no format to validate.

### Three Primitives
1. **Choice** — "which of these 255 categories?" → answer + full ranking
2. **Score** — "how severe, 0 to 10?" → any value, even 7.3
3. **Noul** — "is this urgent?" → 0.87 (a probability, not a guess)

### Quick Start
No API key. No account. No cloud. MIT licensed. Running in under 2 minutes.

---

## USAGE AND MODELS

### Page Description
No accounts, no billing dashboards, no "please try again later."
Download the model once and it's yours — running wherever you want,
answering whenever you ask. Everything runs where you already work:
a local GPU, Ollama on your laptop, or LM Studio on your desktop.

### Decision Prompt Section
One template. Temperature zero. Read the first letter. That's the whole
integration — and it works identically on llama-server, Ollama, and
LM Studio.

### Platform Guides
- **llama-server**: for when you want full control. Your GPU, your
  rules, full probability distributions via n_probs, and speculative
  decoding when you need to generate text too.
- **Ollama**: for when you want it to just work. `ollama create decidex`
  and you're done. It runs alongside your other models.
- **LM Studio**: for when you'd rather click than type. Import the GGUF
  file, hit Start Server, and you have an OpenAI-compatible endpoint
  on localhost.

### Available GGUF Builds
Same model, three sizes — pick the one that fits your VRAM:
- Q4_K_M (5.0 GB) — 16GB GPUs
- Q6_K (6.7 GB) — quality-first
- Q8_0 (8.7 GB) — closest to BF16 behavior
Start small, upgrade later — your code doesn't change.

### Model Downloads
Every adapter, every quantized build, and the full 17,954-sample
training dataset are on HuggingFace under MIT license. Download them
once, own them.

---

## PERFORMANCE

### Page Description
Every number on this page is reproducible — the comparison corpus,
official answers, all 17,954 training samples, and the exact commands
are committed to the repository. Clone it, run the benchmarks, and
check us. No cherry-picking, no hidden benchmarks, no trust required.

### Cross-Source Latency Note
Decidex runs on your GPU; the official API is a remote service. The
table below shows both so you can judge the trade-off: your data never
leaves the machine vs. a network call, zero per-request cost vs.
metered, no rate limit vs. capped.

### Optimization Campaign
We trained five times, measured every round, and kept what worked.
The final adapter (core-8b) agrees with the official API on 98.1% of
yes/no decisions and 100% of multi-choice selections, with probability
MAE of 0.071 on the 86-question comparison corpus.

### Fan-out Scaling
Ask one question or thirty — the response time barely changes. The
model reads your document once and answers everything in parallel from
the same KV cache.

### Optimization Techniques
- **KV Prefix Reuse**: one pass to read, infinite passes to answer.
  11.4× faster on long documents.
- **Cross-request LRU Cache**: the second time you ask about a document,
  it takes 193ms instead of 2.2 seconds. Your users notice.
- **Official-API Distillation**: we didn't guess what the official model
  would say — we asked it thousands of times across four generation
  rounds (via fan-out) and trained on the 17,954 actual answers. Not synthetic, not
  community data — the official model's real outputs, used as the teacher.
  Total data cost: $0.35.

---

## COMPATIBILITY

### Page Description
Already have code that calls the Jev API? Point it at localhost.
That's the migration. No SDK fork, no code rewrite, no vendor lock-in.

### SDK Verification Section
Both official SDKs — Python (typesafe-sdk) and JavaScript
(@typesafe-ai/sdk) — pass all 8 compatibility checks, including typed
answers, grouped accessors, dict questions, error paths, and model
listing. Your code doesn't need to know anything changed.

### Contract Details Table
Field-by-field match with the official OpenAPI contract, verified
against the schema embedded in the official SDK. Every error code
(401/422/429/529), every formula (confidence, score), every limit
(choice ≤255, score 1–26) — implemented the same way.

---

## TRAINING

### Page Description
Most open projects give you the weights and call it a day. We give
you the weights, the training data, the evaluation harness, and the
exact commands to reproduce every number on this site. The only public
model lineage trained on the official API's actual outputs — not
synthetic labels, not community data. Fork it, audit it, rebuild it.

### Pipeline Description
Four steps, all reproducible from the repo:
1. Call the official API across four generation rounds to build the
   training set (fan-out gives 17,954 samples; total cost: $0.35)
2. Train a LoRA adapter (r=32 on Qwen3-8B, frozen base) — about 80
   minutes on a modern GPU
3. Evaluate against committed official answers — 2 minutes, no API
   calls needed
4. Export to GGUF and serve anywhere

### Reproducibility Section
Clone the repo. Run the commands. Get the same numbers. That's the bar
we set for ourselves. If you can't, that's a bug and we'll fix it.

# Decidex Website Copy — Version B (10/10 target)

## HOME

### Hero Badge
FREE · PRIVATE · FOREVER

### Headline
Your GPU. Your data. Your decisions.

### Hero Description
Decidex is a decision model that runs entirely on your own hardware.
It reads your input, picks from your options, and returns a typed
answer with a probability — in under 50 milliseconds, without a single
byte leaving your machine. If you can run a game, you can run this.

### Key Metrics Cards
1. **49ms** — Response Time — "faster than a screen refresh, on hardware you already own"
2. **100%** — API Compatible — "swap one URL in your existing code, done"
3. **97.7%** — Match Rate — "measured against the official API, results in the repo"
4. **5.0 GB** — Download Size — "smaller than most games, runs forever at zero cost"

### Architecture Description
Think of it as a very fast, very honest if-statement. You give it
context and a set of choices. It runs one pass through a language model,
reads the probability of each option, and hands you the answer with a
confidence score. No prose to parse, no format to validate.

### Three Primitives
1. **Choice** — "which of these 255 categories?" → answer + full ranking
2. **Score** — "how severe, 0 to 10?" → any value, even 7.3
3. **Noul** — "is this urgent?" → 0.87 (a probability, not a guess)

### Quick Start
Three lines. No signup. It's running before your coffee is ready.

---

## USAGE AND MODELS

### Page Description
No accounts, no billing dashboards, no "please try again later."
Download the model once, and it's yours — running wherever you want,
answering whenever you ask, for free, forever.

### Decision Prompt Section
One template. Temperature zero. Read the first letter. That's the whole
integration — and it works identically on llama-server, Ollama, and
LM Studio.

### Platform Guides
- **llama-server**: for when you want full control. Your GPU, your
  rules, full probability distributions, and speculative decoding when
  you need to generate text too.
- **Ollama**: for when you want it to just work. `ollama create decidex`
  and you're done. It runs alongside your other models.
- **LM Studio**: for when you'd rather click than type. Import the file,
  hit start, get an OpenAI-compatible endpoint on localhost.

### Available GGUF Builds
Same model, three sizes. Q4_K_M if you have 16GB. Q6_K if you have
room. Q8_0 if you want maximum fidelity. Start small, upgrade later —
your code doesn't change.

### Model Downloads
Everything is on HuggingFace. Adapters, quantized builds, and the full
17,954-sample training dataset. Take what you need.

---

## PERFORMANCE

### Page Description
We measured everything against the official API, then committed all
the data to the repo so you can verify every claim on this page.
No cherry-picking, no hidden benchmarks, no trust required.

### Cross-Source Latency Note
Here's the honest comparison: Decidex on your GPU has zero network
latency, zero per-call cost, and zero rate limits. The official API
has all three. The numbers below let you decide which trade-offs
matter for your use case.

### Optimization Campaign
We trained five times, measured every round, and kept what worked.
The result: 98% agreement on yes/no decisions, probability errors
under 7 points, and a training pipeline you can run yourself.

### Fan-out Scaling
Ask one question or thirty — the response time barely changes. The
model reads your document once and answers everything in parallel.

### Optimization Techniques
- **KV Prefix Reuse**: one pass to read, infinite passes to answer.
  11× faster on long documents.
- **Cross-request Caching**: the second time you ask about a document,
  it takes 193ms instead of 2.2 seconds. Your users notice.
- **Official-API Distillation**: we didn't guess what the official
  model would say — we asked it 17,954 times and trained on the
  actual answers.

---

## COMPATIBILITY

### Page Description
Already have code that calls the Jev API? Point it at localhost.
That's the migration.

### SDK Verification Section
Both official SDKs, every feature, every error path — all tested,
all passing. Your existing code doesn't need to know anything changed.

### Contract Details Table
Field-by-field match with the official OpenAPI contract, verified
against the schema embedded in the official SDK. If the official API
does it, Decidex does it the same way.

---

## TRAINING

### Page Description
Most "open" projects give you the weights and call it a day. We give
you the weights, the training data, the evaluation harness, and the
exact commands to reproduce every number on this site. One GPU,
about 80 minutes, done.

### Pipeline Description
1. Call the official API 1,300 times to build the training set
   (total cost: less than a coffee)
2. Train a LoRA adapter — about 80 minutes on a modern GPU
3. Evaluate against committed official answers — no API calls needed
4. Export to GGUF, deploy anywhere

### Reproducibility Section
The bar: clone the repo, run the commands, get the same numbers. If
you can't, that's a bug and we'll fix it.

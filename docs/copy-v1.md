# Decidex Website Copy — Version A (10/10 target)

## HOME

### Hero Badge
OPEN SOURCE · RUNS ON YOUR GPU · API-COMPATIBLE

### Headline
State in. Typed decisions out.
One forward pass.

### Hero Description
Decidex is an open reimplementation of the Jev decision model that runs
entirely on your machine. Your data never leaves your GPU. No API key to
buy, no rate limit to hit, no cloud to trust. Clone the repo, run one
command, and you have a decision engine that answers in under 50
milliseconds.

### Key Metrics Cards
1. **49ms** — Latency — "on your GPU, every time — no network, no queue"
2. **100%** — Drop-in Compat — "change one URL, both official SDKs just work"
3. **97.7%** — Agreement — "independently measured against the official API"
4. **5.0 GB** — Smallest Build — "fits a gaming laptop, runs offline forever"

### Architecture Description
Your code sends a state and typed questions. Decidex reads the options in
a single forward pass through a frozen language model, and returns typed
answers with calibrated probabilities. No text is generated. Nothing needs
parsing. The answer is already the type your code expects.

### Three Primitives
1. **Choice** — pick one from up to 255 options, get the full ranking
2. **Score** — rate on any scale, the answer can land between levels
3. **Noul** — yes or no as a probability you can branch on

### Quick Start
No API key. No account. No cloud. Just clone and run.

---

## USAGE AND MODELS

### Page Description
Everything runs where you already work — a local GPU, Ollama on your
laptop, or LM Studio on your desktop. No API keys to manage, no rate
limits to dodge, no data leaving your machine.

### Decision Prompt Section
This is the only template you need. Set temperature to zero, ask for one
token, and read the letter. That's the entire integration — whether
you're on llama-server, Ollama, or LM Studio.

### Platform Guides
- **llama-server**: The power-user option. Runs on your GPU, gives you
  full probability distributions, and supports the MTP draft for fast
  generation when you need it.
- **Ollama**: The simplest path. One command to import, one command to
  ask. If you can run `ollama run`, you can run Decidex.
- **LM Studio**: The GUI option. Import the GGUF file, click Start Server,
  and you have an OpenAI-compatible endpoint on localhost.

### Available GGUF Builds
Pick the size that fits your hardware. All three are the same model —
the only difference is precision and file size. Start with Q4_K_M; if
you have 16GB of VRAM, you're done.

### Model Downloads
Every adapter, every quantized build, and the full training dataset are
on HuggingFace. Download them once, own them forever.

---

## PERFORMANCE

### Page Description
Every number on this page is reproducible. The comparison corpus, the
official answers we measured against, and all 17,954 training samples
are committed to the repository. Clone it, run the benchmarks, and
check us.

### Cross-Source Latency Note
Decidex runs on your GPU. The official API is a remote service. The table
shows both so you can judge the trade-off: your data never leaves the
machine vs. a network call, zero per-request cost vs. metered, no rate
limit vs. capped.

### Optimization Campaign
Five rounds of training, each one measured. The final adapter agrees
with the official API on 98% of yes/no decisions, with probability
errors under 7 percentage points.

### Fan-out Scaling
Ask 30 questions in one request. It takes barely longer than asking one.
The state is read once, every question is answered in parallel from the
same cache.

### Optimization Techniques
- **KV Prefix Reuse**: read the state once, answer everything from that
  cache. 11× faster on long documents.
- **Cross-request Caching**: ask the same document again, skip the
  prefill entirely. Cold to warm: 2.2 seconds to 193 milliseconds.
- **Official-API Distillation**: trained on 17,954 real answers from the
  official Jev API. Not synthetic, not community data — the actual
  outputs, used as the teacher.

---

## COMPATIBILITY

### Page Description
Your existing code works. Change one URL, and the official Python or
JavaScript SDK connects to your local Decidex server. No SDK fork, no
code rewrite, no vendor lock-in.

### SDK Verification Section
We tested both official SDKs — every feature, every error path, every
edge case. All eight checks pass. Your code doesn't need to know the
difference.

### Contract Details Table
Every field, every error code, every formula matches the official
OpenAPI contract. We verified this by extracting the schema from the
official SDK and implementing it field by field.

---

## TRAINING

### Page Description
This is the only public model lineage trained on the official API's
actual outputs. Everything is open — the data, the training script, the
evaluation harness. Fork it, audit it, or rebuild it from scratch on a
single GPU.

### Pipeline Description
Four steps, all reproducible from the repo:
1. Generate data from the official API (under $0.35 total)
2. Train the adapter (about 80 minutes on a modern GPU)
3. Evaluate against the committed official answers (2 minutes, no API
   calls needed)
4. Export to GGUF and serve anywhere

### Reproducibility Section
Clone the repo. Run the commands. Get the same numbers. That's the bar
we set for ourselves — and the one you should hold us to.

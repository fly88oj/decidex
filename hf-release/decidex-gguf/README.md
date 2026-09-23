---
license: mit
language: [en, zh, ja]
tags: [gguf, decision-model, system-one, jev, llama.cpp, ollama, lm-studio]
base_model: Qwen/Qwen3-8B
---

# Decidex GGUF builds — decision engine for llama.cpp / Ollama / LM Studio

Quantized builds of the [decidex-core-8b](https://huggingface.co/fly88oj/decidex-core-8b)
adapter merged into Qwen3-8B:

Quantized builds carry a lineage marker (`r1` = first release of the
`core-8b` adapter line, internal training round 7).

| File | Size | Best for |
|---|---|---|
| `decidex-core-8b-r1-Q4_K_M.gguf` | 5.0 GB | 16GB GPUs / 16GB+ RAM CPU |
| `decidex-core-8b-r1-Q6_K.gguf` | 6.7 GB | quality-first |
| `decidex-core-8b-r1-Q8_0.gguf` | 8.7 GB | closest to BF16 behavior |

Companion MTP draft: [decidex-draft-0.6b](https://huggingface.co/fly88oj/decidex-draft-0.6b).

## How to use (decision engine, not chat)

Send the completion prompt below, **temperature 0**, 1–4 max tokens, and
read the first letter — or read letter logprobs for the full probability
distribution (llama-server `n_probs`). Do NOT use a chat template (the
base model is a thinking hybrid; chat templates break the readout).

```
You are a precise decision engine. Evaluate the STATE against the
QUESTION and pick the single best option.

STATE:
{your state}

QUESTION:
{your question}

OPTIONS:
A. {option A}
B. {option B}
C. {option C}

Answer with the letter of the single best option.
Answer:
```

Verified live through a CPU llama-server: official-docs example returns
noul **0.9503** (official 0.95) and score **1.0404** (official 1.05).
Ready-made client: `examples/gguf_decision_client.py` in the
[Decidex repo](https://github.com/fly88oj/decidex); full guide:
`GGUF-USAGE.md` there. Not affiliated with TypeSafe AI.

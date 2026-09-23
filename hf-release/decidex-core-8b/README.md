---
license: mit
language:
  - en
  - zh
  - ja
tags:
  - decision-model
  - system-one
  - jev
  - typesafe
  - structured-output
  - logits
  - lora
base_model: Qwen/Qwen3-8B
library_name: peft
---

# Decidex v7 — the flagship adapter (84/86 official agreement)

LoRA adapter (r=32) for **Qwen/Qwen3-8B** that turns a frozen chat model
into a Jev-style **decision engine**: state in, typed decisions with
probabilities out, one forward pass, zero generated tokens.

Distilled from **17,954 samples of the official Jev API's actual outputs**
(four generation + active-mining rounds; corpus and pipeline in the
[Decidex repo](https://github.com/fly88oj/decidex), see
`REPRODUCE.md`). This is the only public model lineage trained toward the
official model's real answers rather than synthetic labels.

## Measured agreement with the official API

On the 87-question comparison corpus (official answers collected live):

| Primitive | Result |
|---|---|
| Choice top-1 | **23/23 (100%)**, distribution JS divergence 0.0025 |
| Noul decisions | **51/52 (98.1%)** |
| Score modal level | 10/11 (0.909) |
| Overall | **84/86 (97.7%)** |

Generation-distribution disagreement vs the official model: **4.6%**
(mining-round measurement; the 4B baseline was 26%).

## Usage

Serve through the Decidex service (byte-compatible with the official API —
both official SDKs verified):

```bash
pip install -e '.[all]'
decidex serve --engine llm --model Qwen/Qwen3-8B \
  --lora <this-adapter> --device cuda:0
```

Or use GGUF builds of this adapter (merged) for llama.cpp / Ollama /
LM Studio — see the `decidex-gguf` repo.

## Training

- Base: Qwen/Qwen3-8B (frozen, BF16)
- LoRA r=32 α=64 on q/v/o_proj — 0.24% trainable
- Soft-label cross-entropy on the letter-logit readout position
- Dataset: `distill_dataset_v4.jsonl` (7,249 samples = broad sweep +
  score-heavy + first active-mining round ×2), 2 epochs, bs 4 × accum 2
- Hardware: one RTX 4090 24GB, ~80 min

## Honest notes

- Probabilities are distilled, not RLCD-trained; residual disagreements
  with the official model (~2-11% depending on primitive) are semantic
  and documented in `COMPARISON.md` of the repo.
- Not affiliated with TypeSafe AI; Jev and TypeSafe are their trademarks.

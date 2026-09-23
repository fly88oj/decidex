---
license: mit
language: [en, zh, ja]
tags: [decision-model, system-one, jev, logits, lora]
base_model: Qwen/Qwen3-8B
library_name: peft
---

# Decidex v8 — the noul-perfect variant (52/52)

Same recipe as [decidex-core-8b](https://huggingface.co/fly88oj/decidex-core-8b)
but trained on the **de-skewed corpus** (`distill_dataset_v8.jsonl`,
16,220 samples: balanced 50/25/25 kinds, choice arities 2–26, score levels
2/3/10, 24% structured states, 8 domains, ~12% CJK).

| Primitive | v7 | **v8** |
|---|---|---|
| Noul decisions | 51/52 (98.1%) | **52/52 (100%)** |
| Noul probability MAE | 0.0714 | **0.0609** |
| Choice top-1 | 23/23 | 22/23 |
| Score modal | 10/11 | 9/11 |

Choose v8 when noul judgments dominate your workload; choose v7 for the
best overall balance. Full trade-off table: `COMPARISON.md` §10 in the
[Decidex repo](https://github.com/fly88oj/decidex).

Usage identical to v7 (`--lora <this-adapter>`). Not affiliated with
TypeSafe AI.

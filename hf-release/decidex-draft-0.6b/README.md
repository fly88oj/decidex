---
license: mit
tags: [decision-model, speculative-decoding, draft-model, mtp, jev]
base_model: Qwen/Qwen3-0.6B
library_name: peft
---

# Decidex draft 0.6B — the MTP/speculative-decoding companion

Qwen3-0.6B + LoRA (r=16) distilled on the same decision corpus, merged and
exported as GGUF **Q8_0**. Its letter distribution matches the 8B targets,
so it makes a high-acceptance draft for llama.cpp speculative decoding:

```bash
llama-server -m decidex-core-8b-Q4_K_M.gguf \
  --model-draft decidex-draft-0.6b-Q8_0.gguf \
  --spec-draft-n-max 4 -ngl 99
```

Note: speculative decoding accelerates *generation*; the decision readout
itself generates zero tokens. See `GGUF-USAGE.md` in the
[Decidex repo](https://github.com/fly88oj/decidex). Not affiliated
with TypeSafe AI.

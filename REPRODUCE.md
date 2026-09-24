# Full reproduction guide

Everything in the published results (agreement numbers, adapters, GGUF
builds) can be regenerated from this repository. Hardware used: one RTX
4090 24GB (training) + one RTX 5080 16GB (serving/eval); a single 24GB GPU
covers the whole pipeline.

## 0. Environment

```bash
# Rebuild training sets from committed base corpora:
# core = base + score-heavy + active-mining x2
cat benchmarks/distill_dataset.jsonl benchmarks/distill_dataset_score.jsonl     benchmarks/distill_dataset_active.jsonl     benchmarks/distill_dataset_active.jsonl > /tmp/train-core.jsonl

# true = core + balanced-v2
cat /tmp/train-core.jsonl benchmarks/distill_dataset_v2.jsonl > /tmp/train-true.jsonl

# decidex-core-8b (flagship, 84/86)
HF_HOME=... DECIDEX_DEVICE=cuda:0 .venv/Scripts/python benchmarks/distill_train.py   --model <Qwen3-8B> --dtype auto --dataset /tmp/train-core.jsonl   --epochs 2 --batch-size 4 --grad-accum 2 --lora-r 32   --out benchmarks/adapters/decidex-core-8b

# decidex-true-8b (noul-perfect, 52/52): same command with /tmp/train-true.jsonl
# decidex-draft-0.6b (MTP): --model Qwen/Qwen3-0.6B --lora-r 16 --batch-size 16
```

Environment variables used throughout:

```bash
export HF_HOME=/path/to/hf-cache           # model cache (large drive)
export DECIDEX_DEVICE=cuda:0               # GPU for eval/serving
```

Artifacts NOT in git (size) and where to get them:

| Artifact | Size | Source |
|---|---|---|
| Base models Qwen3-4B / 0.6B / 8B | 1.2–16 GB | `hf download Qwen/Qwen3-4B` etc., or ModelScope in CN networks |
| LoRA adapters (v1–v9, draft) | 15–95 MB each | `python scripts/fetch_models.py` (HuggingFace) or retrain per §3 |
| Official-API targets | — | requires your own `TYPESAFE_API_KEY` in `.env` (OpenRouter or TypeSafe) |

Everything else — distillation datasets (26 MB), comparison corpus,
benchmarks, this guide — is in the repository.

## 1. Verify the API contract layer (no GPU, ~5 s)

```bash
.venv/Scripts/python -m pytest tests -m "not slow"   # 60 tests
.venv/Scripts/python -m decidex demo --engine stub   # end-to-end smoke
```

## 2. Base-model evaluation & the official comparison

```bash
# accuracy/latency benchmark on any GPU
HF_HOME=... DECIDEX_DEVICE=cuda:0 .venv/Scripts/python benchmarks/bench.py \
    --tag baseline --json benchmarks/results.json

# real comparison against the official API (needs TYPESAFE_API_KEY)
.venv/Scripts/python benchmarks/compare_official.py            # full corpus
.venv/Scripts/python benchmarks/compare_official.py --self-test  # harness check
```

## 3. Distillation dataset generation

Four generation rounds reproduce the full 17,954-sample corpus (already
committed under `benchmarks/distill_dataset*.jsonl`; regenerate with):

```bash
# round 1 — broad sweep (3186 samples)
.venv/Scripts/python benchmarks/distill_generate.py --states 800
# score-heavy batch (+2219)
.venv/Scripts/python benchmarks/distill_generate.py --states 500 \
    --score-heavy --seed 777 --out benchmarks/distill_dataset_score.jsonl
# balanced v2 batch (+8971; fixes the audited skew)
.venv/Scripts/python benchmarks/distill_generate_v2.py --requests 3000
# active-mining rounds (mine disagreements with any engine)
.venv/Scripts/python benchmarks/distill_active.py --states 500 --seed 333777 \
    --model <base> --lora <adapter>   # engine-configurable
```

Cost of all four rounds at official pricing: **< $0.35** (measured).

## 4. Adapter training (the released lineage)

Training script: `benchmarks/distill_train.py`. Command lines that
produced the released adapters:

```bash
# decidex-core-8b (lineage r1; internal round v7) — flagship (84/86): 8B BF16, LoRA r=32
HF_HOME=... DECIDEX_DEVICE=cuda:0 .venv/Scripts/python benchmarks/distill_train.py \
  --model <Qwen3-8B> --dtype auto --dataset /tmp/train-core.jsonl \
  --epochs 2 --batch-size 4 --grad-accum 2 --lora-r 32 \
  --out benchmarks/adapters/decidex-core-8b

# decidex-true-8b (lineage r1; internal round v8) — noul-perfect (52/52):
#      same command with /tmp/train-true.jsonl
#      (adds the balanced batch; train at bs 2 — wider prompts)
# decidex-draft-0.6b — MTP/0.6B: --model Qwen/Qwen3-0.6B --lora-r 16 --batch-size 16
```

Practical notes learned the hard way (see OPTIMIZATION.md): 8B BF16 needs
bs ≤ 4 + grad-accum on 24GB; 16GB GPUs must train through int4 QLoRA
(bs 2 + gradient checkpointing) — quality-relevant settings are in
`distill_train.py`'s QLoRA path.

## 5. Evaluation of an adapter (against stored official answers)

The official answers for the 86-question scored comparison corpus are committed
(`benchmarks/comparison_raw.json`), so adapter evaluation needs no API:

```bash
# see benchmarks/diagnose_flips.py <adapter> for the residual-flip listing,
# and COMPARISON.md §7–§10 for the exact metric code paths
```

## 6. GGUF + MTP packaging

```bash
# merge adapter into the base
.venv/Scripts/python -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('<Qwen3-8B>', torch_dtype=torch.bfloat16, device_map='cuda:0')
m = PeftModel.from_pretrained(m, 'benchmarks/adapters/decidex-core-8b').merge_and_unload()
m.save_pretrained('hf-release/decidex-core-8b-merged')
AutoTokenizer.from_pretrained('<Qwen3-8B>').save_pretrained('hf-release/decidex-core-8b-merged')"

# convert + quantize (llama.cpp)
python <llama.cpp>/convert_hf_to_gguf.py hf-release/decidex-core-8b-merged \
    --outfile gguf/decidex-core-8b-r1-f16.gguf --outtype f16
llama-quantize gguf/decidex-core-8b-r1-f16.gguf gguf/decidex-core-8b-r1-Q4_K_M.gguf Q4_K_M
# same for Q6_K / Q8_0 and the 0.6B draft (Q8_0)
```

## 7. Publishing to HuggingFace

```bash
export HF_TOKEN=hf_...                   # token with repo.write
python scripts/upload_hf.py              # --skip-gguf to skip the ~20GB repo
```

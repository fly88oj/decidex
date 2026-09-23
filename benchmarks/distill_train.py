"""LoRA distillation: teach Qwen3-4B's letter readout to match official Jev.

Training pairs come from benchmarks/distill_dataset.jsonl (official target
distributions over our exact option texts). The loss is soft-label cross
entropy on the same last-position letter logits the engine reads at
inference — no generation, pure behavior distillation.

    HF_HOME=... DECIDEX_DEVICE=cuda:1 python benchmarks/distill_train.py \
        [--epochs 3] [--batch-size 8] [--lr 1e-4] [--out benchmarks/lora_adapter]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(BENCH / "distill_dataset.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dtype", default="auto", choices=["auto", "int8", "int4"],
                        help="int8/int4 = QLoRA-style training over a quantized base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(BENCH / "lora_adapter"))
    args = parser.parse_args()

    import os

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(BENCH))
    from decidex.engines.llm_logits import LETTERS, LLMLogitsEngine

    device = os.environ.get("DECIDEX_DEVICE", "cuda:1")
    records = [json.loads(line) for line in
               Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"dataset: {len(records)} samples "
          f"({sum(1 for r in records if r['kind']=='noul')} noul / "
          f"{sum(1 for r in records if r['kind']=='score')} score / "
          f"{sum(1 for r in records if r['kind']=='choice')} choice)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs = {"device_map": device}
    if args.dtype in ("int8", "int4"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if args.dtype == "int8"
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_quant_type="nf4"))
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    if args.dtype in ("int8", "int4"):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # letter -> candidate token ids (same variant logic as the engine)
    label_ids: dict[str, list[int]] = {}
    for letter in LETTERS:
        ids = set()
        for variant in (letter, " " + letter):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
        label_ids[letter] = sorted(ids)

    # prompt formatting without loading model weights
    class _P(LLMLogitsEngine):
        def __init__(self):
            self.prompt_variant = "plain"

    samples = []
    for rec in records:
        prompt = _P().build_prompt(rec["state"], rec["instructions"] or "", rec["options"])
        samples.append((prompt, rec["letters"], rec["target"], rec["kind"]))

    # rebalance: noul dominates the raw corpus (~75%); upsample the scarcer
    # kinds so the adapter does not overfit noul softness at score's expense
    upsample = {"score": 3, "choice": 2, "noul": 1}
    rebalanced = []
    for sample in samples:
        rebalanced.extend([sample] * upsample.get(sample[3], 1))
    samples = rebalanced

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr)
    steps_per_epoch = math.ceil(len(samples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"steps: {total_steps} ({steps_per_epoch}/epoch, bs {args.batch_size})")

    step = 0
    t0 = time.perf_counter()
    running = 0.0
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            prompts = [p for p, _l, _t, _k in batch]
            original_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            tokenizer.padding_side = original_side
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            position_ids = (encoded["attention_mask"].cumsum(-1) - 1).clamp(min=0)

            # only the last position feeds the loss; logits_to_keep=1 avoids
            # materializing (B, L, vocab) logits and their backward buffers
            logits = model(**encoded, position_ids=position_ids,
                           logits_to_keep=1).logits[:, -1, :].float()
            loss = logits.new_zeros(())
            for row, (_prompt, letters, target, _kind) in zip(logits, batch):
                scores = torch.stack([
                    max(row[tid] for tid in label_ids[letter]) for letter in letters
                ])
                log_probs = torch.log_softmax(scores, dim=-1)
                loss = loss - (torch.tensor(target, device=logits.device) * log_probs).sum()
            loss = loss / len(batch) / args.grad_accum

            loss.backward()
            if step % args.grad_accum == args.grad_accum - 1 or start + args.batch_size >= len(samples):
                optimizer.step()
                optimizer.zero_grad()

            running += loss.item()
            step += 1
            if step % 50 == 0:
                rate = step / (time.perf_counter() - t0)
                print(f"  step {step}/{total_steps} loss {running / 50:.4f} "
                      f"({rate:.2f} it/s)")
                running = 0.0
            if step >= total_steps:
                done = True
                break

    model.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()

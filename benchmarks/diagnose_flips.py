"""List the remaining noul decision flips for an adapter vs official targets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fit_to_official import NOUL_TEXT_VARIANTS, capture_base, official_p, question_jobs

from decidex.engines.llm_logits import LLMLogitsEngine

adapter = sys.argv[1] if len(sys.argv) > 1 else None
engine = LLMLogitsEngine(model_name="Qwen/Qwen3-4B", device=None, lora_path=adapter)
records = [r for r in question_jobs() if r["type"] == "noul"]
base = capture_base(engine, records, NOUL_TEXT_VARIANTS["bare"])
flips = 0
for r in records:
    p, t = base[f"{r['rid']}.{r['qid']}"][0], official_p(r)[0]
    if (p > 0.5) != (t > 0.5):
        flips += 1
        instr = r["question"].get("instructions") or ""
        print(f"FLIP off={t:.2f} loc={p:.2f} [{r['rid']}.{r['qid']}] "
              f"{instr[:70]}")
        print(f"      state: {str(r['state'])[:90]}")
print(f"\ntotal flips: {flips}/{len(records)}")

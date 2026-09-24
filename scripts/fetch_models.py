"""Fetch the released Decidex artifacts from HuggingFace into this repo.

    DECIDEX_HF_USER=<username> python scripts/fetch_models.py [--which core|true|draft|all]

Downloads the LoRA adapters (v7 / v8 / draft) into benchmarks/ so the
service can load them with --lora. GGUF builds are optional (--gguf).
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ADAPTERS = {
    "core": ("decidex-core-8b", "adapters/decidex-core-8b"),
    "true": ("decidex-true-8b", "adapters/decidex-true-8b"),
    "draft": ("decidex-draft-0.6b", None),  # GGUF only
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=os.environ.get("DECIDEX_HF_USER", ""))
    parser.add_argument("--which", choices=list(ADAPTERS) + ["all"], default="all")
    parser.add_argument("--gguf", action="store_true", help="also fetch GGUF builds")
    args = parser.parse_args()

    user = args.user or "fly88oj"
    print(f"downloading from huggingface.co/{user}/ ...")

    wanted = list(ADAPTERS) if args.which == "all" else [args.which]
    for name in wanted:
        repo, target_dir = ADAPTERS[name]
        target = ROOT / "benchmarks" / (target_dir or f"gguf_{repo}")
        if target.exists() and any(target.iterdir()):
            print(f"  {name}: already present ({target})")
            continue
        print(f"  {name}: -> {target}")
        subprocess.run(["hf", "download", f"{user}/{repo}",
                        "--local-dir", str(target)], check=True)

    if args.gguf:
        target = ROOT / "gguf"
        target.mkdir(exist_ok=True)
        subprocess.run(["hf", "download", f"{user}/decidex-gguf",
                        "--local-dir", str(target)], check=True)

    print("done. serve with: decidex serve --engine llm --model Qwen/Qwen3-8B "
          "--lora benchmarks/adapters/decidex-core-8b")


if __name__ == "__main__":
    main()

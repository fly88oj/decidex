"""Publish Decidex artifacts to HuggingFace (token auth).

    HF_TOKEN=hf_... python scripts/upload_hf.py [--skip-gguf]

Repos: decidex-core-8b / decidex-true-8b (LoRA adapters),
decidex-draft-0.6b (MTP draft GGUF), decidex-gguf (Q4/Q6/Q8 builds),
decidex-distill-data (full training corpus).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
USER = "fly88oj"


def publish(api: HfApi, repo: str, repo_type: str, folder: Path, path_in_repo: str = "."):
    api.create_repo(f"{USER}/{repo}", repo_type=repo_type, exist_ok=True, private=False)
    api.upload_folder(repo_id=f"{USER}/{repo}", repo_type=repo_type,
                      folder_path=str(folder), path_in_repo=path_in_repo,
                      commit_message=f"Release Decidex artifacts ({repo})")
    print(f"done: https://huggingface.co/{USER}/{repo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gguf", action="store_true",
                        help="skip the ~20GB quantized-build repo")
    parser.add_argument("--only", default=None,
                        help="upload a single repo (core|true|draft|gguf|data)")
    args = parser.parse_args()

    import os
    api = HfApi(token=os.environ.get("HF_TOKEN", ""))

    hf = ROOT / "hf-release"

    def copy_adapter(src: str, dst: Path):
        dst.mkdir(parents=True, exist_ok=True)
        target = dst / "adapter"
        if not target.exists():
            shutil.copytree(ROOT / "benchmarks" / src, target)

    if args.only in (None, "core"):
        copy_adapter("adapters/decidex-core-8b", hf / "decidex-core-8b")
        publish(api, "decidex-core-8b", "model", hf / "decidex-core-8b")
    if args.only in (None, "true"):
        copy_adapter("adapters/decidex-true-8b", hf / "decidex-true-8b")
        publish(api, "decidex-true-8b", "model", hf / "decidex-true-8b")
    if args.only in (None, "draft"):
        draft = hf / "decidex-draft-0.6b"
        draft.mkdir(exist_ok=True)
        for src in (ROOT / "gguf").glob("decidex-draft-0.6b-*.gguf"):
            shutil.copy(src, draft / src.name)
        publish(api, "decidex-draft-0.6b", "model", draft)
    if args.only == "data":
        data = hf / "decidex-distill-data"
        data.mkdir(exist_ok=True)
        for src in (ROOT / "benchmarks").glob("distill_dataset*.jsonl"):
            shutil.copy(src, data / src.name)
        publish(api, "decidex-distill-data", "dataset", data)
    if args.only in (None, "gguf") and not args.skip_gguf:
        gguf = hf / "decidex-gguf"
        gguf.mkdir(exist_ok=True)
        for pattern in ("decidex-core-8b-r1-Q*.gguf",):
            for src in (ROOT / "gguf").glob(pattern):
                shutil.copy(src, gguf / src.name)
        publish(api, "decidex-gguf", "model", gguf)


if __name__ == "__main__":
    main()

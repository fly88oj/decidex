# Contributing to Decidex

Thanks for your interest in improving Decidex. This document covers the
essentials; when in doubt, open an issue first and ask.

## Getting started

```bash
git clone <your-fork-url> && cd Decidex
python -m venv .venv

# Fast path: API contract layer only (no ML stack needed)
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest tests -m "not slow"

# Full path: adds torch/transformers/sentence-transformers
.venv/Scripts/python -m pip install -e ".[all,dev]"
```

Slow tests need a GPU (or patience on CPU) and download models on first run:

```bash
HF_HOME=/path/to/large-drive/hf-cache DECIDEX_DEVICE=cuda:1 \
  .venv/Scripts/python -m pytest tests -m slow
```

## Project layout

| Path | Purpose |
|---|---|
| `decidex/server.py` | FastAPI app: request validation, parallel evaluation, official response shapes |
| `decidex/sdk.py` | Client SDK mirroring the official `typesafe_sdk` |
| `decidex/calib.py` | Probability math with the exact official semantics (confidence, weighted score) |
| `decidex/render.py` | `state` / `instructions` / `criteria` → text for the engines |
| `decidex/cli.py` | `serve` / `demo` entry points |
| `decidex/engines/` | Scoring engines: `llm_logits` (default), `embedding`, `stub` |
| `tests/` | Fast contract tests + `@pytest.mark.slow` engine tests |
| `examples/` | End-to-end usage examples |

## Design rules

These keep the project faithful to what it replicates — breaking one of
these is almost certainly a bug:

1. **API compatibility is a contract.** Request/response fields, error
   statuses (401/422/429/529), limits (Choice ≤ 255 options, Score 2–10
   levels), and the confidence formula must keep matching the official
   documentation (see `RESEARCH.md` for the evidence trail).
2. **Engines answer one question**: given `(state_text, instruction_text,
   option_texts)`, return a distribution over the options. Everything else
   (Noul/Choice/Score shapes, confidence, rounding) lives in the server
   layer. Don't push answer-shape logic into engines.
3. **Questions are independent.** One question must never influence
   another's answer; that is the official parallel-evaluation semantics.
4. **No ML imports at package import time.** `import decidex` must work
   without torch installed; heavy dependencies load lazily inside engine
   constructors.
5. **Every formula needs a source.** If you change math in `calib.py`,
   cite the official doc/example that justifies the new behavior, and keep
   the regression test that pins it.

## Pull requests

- Keep PRs focused; one concern per PR.
- New features need tests. Fast tests must stay dependency-free.
- Bug fixes should include a test that fails before the fix.
- Update `CHANGELOG.md` under an `## [Unreleased]` heading.
- Run the full suite before submitting:

```bash
.venv/Scripts/python -m pytest tests -m "not slow"
.venv/Scripts/python -m pytest tests -m slow   # if you have the ML stack
```

## Documentation

`README.md` is the English base document. `README.zh-CN.md` and
`README.ja.md` are translations — if you change the base README, update the
translations in the same PR or flag that they need updating.

## Trademark notice

This project is not affiliated with TypeSafe AI. "Jev" and "TypeSafe" are
trademarks of their respective owners; Decidex replicates the documented
public API shape for interoperability and local development.

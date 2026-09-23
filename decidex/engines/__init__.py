from __future__ import annotations

from decidex.engines.base import Engine, StubEngine

__all__ = ["Engine", "StubEngine", "build_engine"]


def build_engine(kind: str, model: str | None = None, **kwargs) -> Engine:
    """Instantiate an engine by name: 'llm', 'embedding', or 'stub'.

    Heavy dependencies are imported lazily so the API contract layer (server,
    SDK, tests) runs without any ML stack installed.
    """
    if kind == "stub":
        return StubEngine(**kwargs)
    if kind == "embedding":
        from decidex.engines.embedding import EmbeddingEngine

        return EmbeddingEngine(model_name=model, **kwargs)
    if kind == "llm":
        from decidex.engines.llm_logits import LLMLogitsEngine

        return LLMLogitsEngine(model_name=model, **kwargs)
    raise ValueError(f"unknown engine kind: {kind!r} (expected 'llm', 'embedding', or 'stub')")

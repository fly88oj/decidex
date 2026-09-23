"""Embedding engine: semantic similarity as a dependency-light fallback.

Embeds the state once and each option hypothesis with a sentence-transformer,
then converts cosine similarities into a probability distribution through a
temperature softmax (default temperature 0.05, since cosine similarities live
in a narrow band around 0.2-0.6).

This is a heuristic engine: it has no notion of judgment beyond lexical/
semantic overlap with the option descriptions, so treat its probabilities as
relative rankings. It exists for CPU-only machines and for tests; the LLM
logits engine is the one that reproduces Jev-like behavior.
"""

from __future__ import annotations

from decidex.calib import softmax
from decidex.engines.base import Engine


class EmbeddingEngine(Engine):
    name = "embedding"

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.05,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        super().__init__(model_id=model_name, temperature=temperature)
        self.model = SentenceTransformer(model_name, device=device)

    def score_batch(self, jobs: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        flat_options: list[str] = []
        counts: list[int] = []
        for _state, _instruction, options in jobs:
            counts.append(len(options))
            flat_options.extend(options)

        state_embeddings = self.model.encode(
            [state for state, _instruction, _options in jobs],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        option_embeddings = self.model.encode(
            flat_options, normalize_embeddings=True, convert_to_numpy=True
        )

        results: list[list[float]] = []
        cursor = 0
        for row, count in zip(state_embeddings, counts):
            sims = option_embeddings[cursor : cursor + count] @ row
            cursor += count
            results.append(softmax(sims.tolist(), temperature=self.temperature))
        return results

"""Engine interface.

An engine answers exactly one question: given the rendered state text, the
rendered instruction text, and N option texts, return a probability
distribution over the options. Noul (2 options: yes/no), Choice, and Score are
all built on this single primitive, mirroring how the official API evaluates
every question independently and in parallel against the same state.
"""

from __future__ import annotations

import re
from typing import Callable

from decidex.calib import softmax
from decidex.render import estimate_tokens

_WORD = re.compile(r"[a-z0-9]+")


def _stems(text: str) -> set[str]:
    """Lowercased word stems (naive plural stripping) for lexical overlap."""
    stems = set()
    for word in _WORD.findall(text.lower()):
        stems.add(word)
        if len(word) > 3 and word.endswith("s"):
            stems.add(word[:-1])
    return stems


class Engine:
    """Base class. Subclasses override ``score_batch``."""

    name = "base"

    def __init__(self, model_id: str = "stub-1", temperature: float = 1.0):
        self.model_id = model_id
        self.temperature = temperature

    def score_batch(
        self, jobs: list[tuple[str, str, list[str]]]
    ) -> list[list[float]]:
        """Score a batch of (state_text, instruction_text, option_texts) jobs.

        Returns one distribution per job (softmax at the engine's temperature,
        entries summing to 1). Engines should evaluate jobs independently, the
        way the official API evaluates questions in parallel.
        """
        return [self.score(state, instruction, options) for state, instruction, options in jobs]

    def score(self, state_text: str, instruction_text: str, option_texts: list[str]) -> list[float]:
        return self.score_batch([(state_text, instruction_text, option_texts)])[0]

    def token_count(self, text: str) -> int:
        """Token count for usage accounting.

        The default is a chars/4 estimate; engines with a real tokenizer
        (e.g. the LLM engine) override this for exact counts.
        """
        return estimate_tokens(text)


def keyword_scores(state_text: str, instruction_text: str, option_texts: list[str]) -> list[float]:
    """Tiny lexical scorer: word-stem overlap between (state + instruction) and each option.

    Deterministic and dependency-free; good enough to exercise the full API
    contract without an ML stack.
    """
    context_stems = _stems(state_text) | _stems(instruction_text)

    def overlap(option: str) -> float:
        return float(len(_stems(option) & context_stems))

    return [overlap(o) for o in option_texts]


class StubEngine(Engine):
    """Deterministic engine for tests and dependency-free local runs.

    Wraps a logits function (default: the lexical ``keyword_scores`` scorer)
    in the standard engine contract.
    """

    def __init__(self, logits_fn: Callable[[str, str, list[str]], list[float]] | None = None,
                 model_id: str = "stub-1", temperature: float = 1.0):
        super().__init__(model_id=model_id, temperature=temperature)
        self.name = "stub"
        self._logits_fn = logits_fn or keyword_scores

    def score_batch(self, jobs: list[tuple[str, str, list[str]]]) -> list[list[float]]:
        return [
            softmax(self._logits_fn(state, instruction, options), temperature=self.temperature)
            for state, instruction, options in jobs
        ]

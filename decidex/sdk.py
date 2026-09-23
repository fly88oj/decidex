"""Decidex client SDK, interface-compatible with the official ``typesafe_sdk``.

Usage mirrors the official SDK exactly (only the import and base URL change):

    from decidex import Choice, DecidexClient, Noul, Score

    client = DecidexClient()  # http://127.0.0.1:8600 by default
    response = client.system_one(
        state={"message": "I was charged twice."},
        questions={
            "category": Choice(instructions="What is this about?",
                               criteria={"billing": "Payment issues", "other": "Anything else"}),
            "urgent": Noul(instructions="The message conveys urgency"),
            "frustration": Score(instructions="How frustrated is the customer",
                                 criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )
    response.answers["category"].choice
    response.answers["frustration"].score
    response.answers["urgent"].noul

Retries on 429/529 with exponential backoff and Retry-After support, matching
the official SDK's documented retry behavior.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from decidex import DEFAULT_BASE_URL, MODEL_ALIAS

RETRYABLE_STATUS = {429, 529}
DEFAULT_TIMEOUT = 120.0


# ---------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------

@dataclass
class Noul:
    instructions: Any
    criteria: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


@dataclass
class Choice:
    instructions: Any
    criteria: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass
class Score:
    instructions: Any
    criteria: list[Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


Question = Noul | Choice | Score | Mapping[str, Any]


def _question_payload(question: Question) -> dict[str, Any]:
    if hasattr(question, "to_payload"):
        return question.to_payload()
    if isinstance(question, Mapping):
        return dict(question)
    raise TypeError(f"question must be a Noul/Choice/Score or a dict, got {type(question)!r}")


# ---------------------------------------------------------------------------
# Typed answers
# ---------------------------------------------------------------------------

@dataclass
class NoulAnswer:
    type: str
    noul: float


@dataclass
class ChoiceAnswer:
    type: str
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class ScoreAnswer:
    type: str
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def _parse_answer(payload: dict[str, Any]) -> Answer:
    answer_type = payload.get("type")
    if answer_type == "noul":
        return NoulAnswer(type="noul", noul=float(payload["noul"]))
    if answer_type == "choice":
        return ChoiceAnswer(
            type="choice",
            choice=str(payload["choice"]),
            probabilities={k: float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    if answer_type == "score":
        return ScoreAnswer(
            type="score",
            score=float(payload["score"]),
            legend=dict(payload["legend"]),
            probabilities={k: float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    raise ValueError(f"unknown answer type: {answer_type!r}")


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class SystemOneResponse:
    model: str
    answers: dict[str, Answer]
    usage: Usage = field(default_factory=lambda: Usage(0, 0))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class DecidexError(Exception):
    def __init__(self, message: str, status_code: int | None = None,
                 field_name: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.field = field_name


class DecidexClient:
    """Synchronous client for a Decidex server (drop-in for TypeSafeClient)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = MODEL_ALIAS,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or os.environ.get("DECIDEX_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("DECIDEX_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self._http = httpx.Client(timeout=timeout)

    def __enter__(self) -> "DecidexClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -- endpoints -----------------------------------------------------------
    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Question],
        model: str | None = None,
    ) -> SystemOneResponse:
        payload = {
            "state": state,
            "model": model or self.model,
            "questions": {qid: _question_payload(q) for qid, q in questions.items()},
        }
        data = self._request("POST", "/v1/systemone", json=payload)
        return SystemOneResponse(
            model=data["model"],
            answers={qid: _parse_answer(a) for qid, a in data["answers"].items()},
            usage=Usage(**data.get("usage", {})),
        )

    def models(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/v1/models")
        return data["models"]

    # -- transport -------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs) -> Any:
        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        attempt = 0
        while True:
            try:
                response = self._http.request(method, self.base_url + path, headers=headers, **kwargs)
            except httpx.TransportError:
                # Connection-level failures (incl. timeouts, which subclass
                # TransportError) are retryable, like the official SDK.
                if attempt >= self.max_retries:
                    raise
                time.sleep(_backoff_seconds(attempt))
                attempt += 1
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                delay = _retry_after_delay(response, attempt)
                time.sleep(delay)
                attempt += 1
                continue
            if response.status_code >= 400:
                message, field = _error_detail(response)
                raise DecidexError(
                    message or f"HTTP {response.status_code}",
                    status_code=response.status_code,
                    field_name=field,
                )
            return response.json()


def _error_detail(response: httpx.Response) -> tuple[str | None, str | None]:
    """Extract (message, field) from an official-style error body.

    422 bodies are FastAPI-shaped: ``{"detail": [{"loc": [...], "msg": ...}]}``;
    other errors use ``{"detail": "<text>"}``. Unknown shapes yield no detail.
    """
    try:
        detail = response.json().get("detail")
    except Exception:
        return None, None
    if isinstance(detail, list) and detail:
        first = detail[0]
        loc = first.get("loc") or []
        field = ".".join(str(part) for part in loc[1:]) if len(loc) > 1 else None
        return first.get("msg"), field
    if isinstance(detail, str):
        return detail, None
    return None, None


def _retry_after_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass  # HTTP-date form: fall through to exponential backoff
    return _backoff_seconds(attempt)


def _backoff_seconds(attempt: int) -> float:
    return min(2.0 ** attempt, 30.0)

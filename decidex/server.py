"""FastAPI service replicating the official TypeSafe System One HTTP API.

Endpoint contract mirrors https://docs.typesafe.ai/api :

    POST /v1/systemone   {state, model, questions} -> {model, answers, usage}
    GET  /v1/models      -> model catalog
    GET  /health         -> liveness + engine info

Questions are validated to the official limits (Choice: <=255 options, Score:
1..26 levels — openapi.json floor 1, letter-readout ceiling 26, official prose
recommends 2..10 — Noul: optional {true, false} criteria), evaluated
independently against the shared state, and answered with the documented
answer shapes: score = sum(i * p_i), confidence = clamp((K*p_max - 1)/(K - 1),
0, 1), probabilities summing to 1. Errors use the official status codes
(401, 422, 429, 529) with a body that names the offending field.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any, NamedTuple

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from decidex import MODEL_ALIAS, MODEL_ID
from decidex.calib import confidence, round_distribution, weighted_score
from decidex.engines.base import Engine
from decidex.render import render_state, render_text

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 1   # official openapi.json: min_length=1 (docs prose recommends 2+)
MAX_SCORE_LEVELS = 26  # letter-readout capacity; official docs prose says up to 10

# Accepted model names. Beyond the Decidex ids, the official Jev ids are
# accepted so unmodified client code written against the real API can be
# pointed at this server by changing only the base URL.
_MODEL_ALIASES = {
    MODEL_ALIAS: MODEL_ID,
    MODEL_ID: MODEL_ID,
    "jev-latest": "jev-1.13.0",
    "jev-1.13.0": "jev-1.13.0",
}


class ApiError(Exception):
    """Error carrying the official FastAPI-style validation body.

    422 responses use ``{"detail": [{"loc": [...], "msg": ..., "type": ...}]}``
    exactly like the official API (its schema is embedded in the official SDK);
    other statuses use ``{"detail": "<message>"}``.
    """

    def __init__(self, status: int, message: str, loc: list[str] | None = None,
                 error_type: str = "value_error"):
        self.status = status
        self.message = message
        self.loc = loc or []
        self.error_type = error_type


def _invalid(message: str, field: str | None = None, error_type: str = "value_error") -> ApiError:
    loc = ["body"]
    if field:
        loc.extend(field.split("."))
    return ApiError(422, message, loc=loc, error_type=error_type)


def _missing(field: str) -> ApiError:
    return _invalid("Field required", field, error_type="missing")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_system_one(body: Any) -> tuple[Any, str, dict]:
    if not isinstance(body, dict):
        raise _invalid("Request body must be a JSON object.")

    if "state" not in body:
        raise _missing("state")
    state = body["state"]
    if not isinstance(state, (str, dict, list)):
        raise _invalid("'state' must be a string, object, or array.", "state")

    model = body.get("model", MODEL_ALIAS)
    if not isinstance(model, str) or not model:
        raise _invalid("'model' must be a non-empty string.", "model")

    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise _invalid("'questions' must be a non-empty map of question id to question.", "questions")

    for qid, question in questions.items():
        prefix = f"questions.{qid}"
        if not isinstance(question, dict):
            raise _invalid("A question must be a JSON object.", prefix)
        qtype = question.get("type")
        if qtype not in ("noul", "choice", "score"):
            raise _invalid("Question 'type' must be one of 'noul', 'choice', 'score'.", f"{prefix}.type")
        # The official openapi.json marks instructions optional on all three
        # question types; a missing/null instruction renders as empty text.
        instructions = question.get("instructions")
        if instructions is not None and not isinstance(instructions, (str, dict, list)):
            raise _invalid("'instructions' must be a string, object, or array.", f"{prefix}.instructions")

        criteria = question.get("criteria")
        if qtype == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise _invalid(
                        "Noul 'criteria' must be an object with optional 'true'/'false'.",
                        f"{prefix}.criteria",
                    )
                for key, value in criteria.items():
                    if key not in ("true", "false"):
                        raise _invalid(
                            f"Unknown noul criteria key {key!r}; expected 'true' or 'false'.",
                            f"{prefix}.criteria",
                        )
                    if value is not None and not isinstance(value, (str, dict, list)):
                        raise _invalid(
                            f"Noul criteria {key!r} must be a string, object, array, or null.",
                            f"{prefix}.criteria.{key}",
                        )
        elif qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise _invalid(
                    "Choice 'criteria' must be a non-empty map of option to description.",
                    f"{prefix}.criteria",
                )
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise _invalid(
                    f"Choice supports at most {MAX_CHOICE_OPTIONS} options; got {len(criteria)}.",
                    f"{prefix}.criteria",
                )
            for option, description in criteria.items():
                if description is not None and not isinstance(description, (str, dict, list)):
                    raise _invalid(
                        f"Choice option {option!r} description must be a string, object, array, or null.",
                        f"{prefix}.criteria.{option}",
                    )
        else:  # score
            if not isinstance(criteria, list) or not (
                MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS
            ):
                raise _invalid(
                    f"Score 'criteria' must be an ordered array of {MIN_SCORE_LEVELS} "
                    f"to {MAX_SCORE_LEVELS} level descriptions.",
                    f"{prefix}.criteria",
                )
            for i, level in enumerate(criteria):
                if not isinstance(level, (str, dict, list)):
                    raise _invalid(
                        f"Score level {i} must be a string, object, or array.",
                        f"{prefix}.criteria.{i}",
                    )

    return state, model, questions


# ---------------------------------------------------------------------------
# Evaluation: question -> option readout -> typed answer
# ---------------------------------------------------------------------------

def _noul_option_texts(question: dict) -> list[str]:
    criteria = question.get("criteria") or {}
    true_desc = criteria.get("true")
    false_desc = criteria.get("false")
    yes = f"yes - {render_text(true_desc)}" if true_desc is not None else "yes"
    no = f"no - {render_text(false_desc)}" if false_desc is not None else "no"
    return [yes, no]


def _choice_option_texts(question: dict) -> tuple[list[str], list[str]]:
    keys = list(question["criteria"].keys())
    texts = []
    for key, description in question["criteria"].items():
        rendered = render_text(description) if description is not None else None
        texts.append(f"{key} - {rendered}" if rendered is not None else str(key))
    return keys, texts


def _score_option_texts(question: dict) -> list[str]:
    return [render_text(level) for level in question["criteria"]]


class Job(NamedTuple):
    """One question prepared for engine evaluation.

    ``choice_keys`` carries the option key order for choice questions (None
    otherwise) so answer assembly never re-renders criteria.
    """

    instruction: str
    options: list[str]
    kind: str
    choice_keys: list[str] | None


def build_jobs(state_text: str, questions: dict) -> dict[str, Job]:
    """Map each question id to its evaluation job."""
    jobs: dict[str, Job] = {}
    for qid, question in questions.items():
        instruction = render_text(question.get("instructions"))
        qtype = question["type"]
        keys: list[str] | None = None
        if qtype == "noul":
            options = _noul_option_texts(question)
        elif qtype == "choice":
            keys, options = _choice_option_texts(question)
        else:
            options = _score_option_texts(question)
        jobs[qid] = Job(instruction, options, qtype, keys)
    return jobs


def evaluate(engine: Engine, state: Any, questions: dict) -> dict[str, Any]:
    state_text = render_state(state)
    jobs = build_jobs(state_text, questions)

    batch = [(state_text, job.instruction, job.options) for job in jobs.values()]
    distributions = engine.score_batch(batch)

    answers: dict[str, Any] = {}
    for (qid, job), probs in zip(jobs.items(), distributions):
        probs = round_distribution(probs)
        if job.kind == "noul":
            answers[qid] = {"type": "noul", "noul": round(probs[0], 4)}
        elif job.kind == "choice":
            keys = job.choice_keys or []
            top = max(range(len(keys)), key=lambda i: probs[i])
            answers[qid] = {
                "type": "choice",
                "choice": keys[top],
                "probabilities": {key: p for key, p in zip(keys, probs)},
                "confidence": round(confidence(probs), 4),
            }
        else:
            # legend echoes the criteria entries verbatim (they may be
            # structured values), matching the official ScoreAnswer schema.
            levels = questions[qid]["criteria"]
            answers[qid] = {
                "type": "score",
                "score": round(weighted_score(probs), 4),
                "legend": {str(i): level for i, level in enumerate(levels)},
                "probabilities": {str(i): p for i, p in enumerate(probs)},
                "confidence": round(confidence(probs), 4),
            }
    return answers


def estimate_usage(engine: Engine, state: Any, questions: dict) -> dict[str, int]:
    """Input-token count over state + questions; output counted per decision.

    Delegates to the engine's token_count (exact when it has a tokenizer,
    chars/4 estimate otherwise). Output tokens are "too cheap to meter" in
    the official API; we report one token per option/level/noul as a stand-in.
    """
    counter = engine.token_count
    texts = [render_state(state)]
    for question in questions.values():
        texts.append(render_text(question.get("instructions")))
        criteria = question.get("criteria")
        if question["type"] == "choice":
            texts.extend(str(k) if d is None else render_text(d) for k, d in criteria.items())
        elif question["type"] == "score":
            texts.extend(render_text(level) for level in criteria)
        else:
            for key in ("true", "false"):
                if criteria and criteria.get(key) is not None:
                    texts.append(render_text(criteria[key]))
    input_tokens = sum(counter(text) for text in texts)

    output_tokens = 0
    for question in questions.values():
        qtype = question["type"]
        if qtype == "choice":
            output_tokens += len(question["criteria"])
        elif qtype == "score":
            output_tokens += len(question["criteria"])
        else:
            output_tokens += 1
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(
    engine: Engine | None = None,
    *,
    engine_factory=None,
    api_key: str | None = None,
) -> FastAPI:
    """Build the service. Either pass a ready ``engine`` or an ``engine_factory``
    (a zero-argument callable) that is invoked once at startup — useful for
    engines whose model download should happen at serve time, not import time.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if _app.state.engine is None and engine_factory is not None:
            _app.state.engine = engine_factory()
        yield

    app = FastAPI(title="Decidex — local System One decision API", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.api_key = api_key if api_key is not None else os.environ.get("DECIDEX_API_KEY")
    # GPU work is serialized by a lock (models are not safe for concurrent
    # forward passes); a slot semaphore bounds how many requests may wait,
    # turning overload into 429s instead of unbounded queues.
    app.state.eval_lock = threading.Lock()
    app.state.eval_slots = threading.Semaphore(int(os.environ.get("DECIDEX_MAX_QUEUE", "8")))

    def error_response(err: ApiError) -> JSONResponse:
        # Official FastAPI-style bodies (the official API's openapi.json is
        # embedded in typesafe-sdk): structured `detail` for 422, plain
        # `detail` string otherwise.
        if err.status == 422:
            body: Any = {"detail": [{"loc": err.loc, "msg": err.message, "type": err.error_type}]}
        else:
            body = {"detail": err.message}
        headers = {"Retry-After": "1"} if err.status in (429, 529) else None
        return JSONResponse(status_code=err.status, content=body, headers=headers)

    @app.middleware("http")
    async def auth_and_errors(request: Request, call_next):
        # /health stays reachable without credentials so monitoring probes
        # keep working when an API key is configured.
        expected = app.state.api_key
        if expected and request.url.path != "/health":
            header = request.headers.get("authorization", "")
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or token.strip() != expected:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key."},
                )
        return await call_next(request)

    @app.exception_handler(ApiError)
    async def api_error_handler(_request: Request, err: ApiError):
        return error_response(err)

    @app.get("/health")
    async def health():
        engine: Engine | None = app.state.engine
        return {
            "status": "ok",
            "engine": engine.name if engine else None,
            "engine_model": engine.model_id if engine else None,
        }

    @app.get("/v1/models")
    async def list_models():
        # Official ModelMetadataList shape: every accepted name with a
        # human-readable description and a release date.
        engine: Engine | None = app.state.engine
        model_entries = [
            {
                "name": MODEL_ALIAS,
                "description": "Decidex flagship alias; resolves to the current Decidex model.",
                "release_date": "2026-09-21",
            },
            {
                "name": MODEL_ID,
                "description": "Decidex's served model: direct typed-logit readout over a frozen open LM.",
                "release_date": "2026-09-21",
            },
        ]
        if engine and engine.model_id not in (MODEL_ID, MODEL_ALIAS):
            model_entries.append(
                {
                    "name": engine.model_id,
                    "description": f"Backing engine model for {MODEL_ALIAS} ({engine.name} engine).",
                    "release_date": "2026-09-21",
                }
            )
        return {"models": model_entries}

    @app.post("/v1/systemone")
    async def system_one(request: Request):
        engine: Engine | None = app.state.engine
        if engine is None:
            return JSONResponse(
                status_code=529,
                content={"detail": "Engine not ready."},
            )
        # Reject oversized bodies before JSON parsing (the official API caps
        # at 64k input tokens; a hard byte limit here is the DoS floor).
        max_body = int(os.environ.get("DECIDEX_MAX_BODY_BYTES", 10 * 1024 * 1024))
        content_length = int(request.headers.get("content-length", 0))
        if content_length > max_body:
            raise ApiError(413, f"Request body too large ({content_length} bytes; limit {max_body}).")
        try:
            body = await request.json()
        except Exception:
            raise _invalid("Request body must be valid JSON.")
        state, model, questions = validate_system_one(body)

        resolved = _MODEL_ALIASES.get(model)
        if resolved is None:
            if engine and model == engine.model_id:
                resolved = model
            else:
                raise _invalid(
                    f"Unknown model {model!r}. Use 'decidex-latest' or an id from GET /v1/models.",
                    "model",
                )

        def run_evaluation() -> tuple[dict, dict]:
            """Slot-guarded, GPU-serialized evaluation + usage accounting.

            Runs in the threadpool so the event loop stays responsive, and
            holds the engine lock so concurrent requests queue instead of
            racing the model.
            """
            slots: threading.Semaphore = app.state.eval_slots
            if not slots.acquire(blocking=False):
                raise ApiError(429, "Too many concurrent evaluations; retry shortly.")
            try:
                with app.state.eval_lock:
                    answers = evaluate(engine, state, questions)
                    usage = estimate_usage(engine, state, questions)
                return answers, usage
            finally:
                slots.release()

        answers, usage = await run_in_threadpool(run_evaluation)

        # Context limit, mirroring the official 64k budget for state+questions.
        # Engines may declare a smaller practical limit (e.g. the LLM engine's
        # max_input_tokens); env var sets the server-wide cap.
        engine_limit = getattr(engine, "max_input_tokens", None)
        server_limit = int(os.environ.get("DECIDEX_MAX_INPUT_TOKENS", 65536))
        limit = min(x for x in (engine_limit, server_limit) if x)
        if usage["input_tokens"] > limit:
            raise _invalid(
                f"Request needs ~{usage['input_tokens']} input tokens; limit is {limit}.",
                "state",
            )

        return {"model": resolved, "answers": answers, "usage": usage}

    return app

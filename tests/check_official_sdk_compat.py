"""Compatibility check: the OFFICIAL typesafe-sdk pointed at the local Decidex server.

Run with the server up:
    HF_HOME=... DECIDEX_DEVICE=cuda:1 python -m decidex serve --engine llm --port 8600
    python tests/check_official_sdk_compat.py
"""

from __future__ import annotations

import sys

from typesafe_sdk import (
    Choice,
    Noul,
    Score,
    TypeSafeClient,
    TypeSafeUnprocessableEntityError,
)

BASE_URL = "http://127.0.0.1:8600"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} - {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> None:
    # The SDK refuses to construct a client without a key; the local server
    # ignores the header when no key is configured, so a dummy key works.
    with TypeSafeClient(api_key="local-dev", base_url=BASE_URL) as client:
        # ---- 1. the official docs' headline example, verbatim style --------
        response = client.system_one(
            state={
                "ticket_message": "My flight was cancelled. Can I get a refund?",
                "refund_policy": "Cancelled flights are eligible for a full refund.",
            },
            questions={
                "refund_requested": Noul(
                    instructions="Does `ticket_message` request a refund?",
                ),
                "request_type": Choice(
                    instructions="What is the main request in `ticket_message`?",
                    criteria={
                        "refund": "The customer wants money returned.",
                        "rebooking": "The customer wants a replacement flight.",
                        "information": "The customer is asking for information only.",
                    },
                ),
                "frustration": Score(
                    instructions="How frustrated does the customer appear in `ticket_message`?",
                    criteria=[
                        "Calm and neutral.",
                        "Concerned but civil.",
                        "Very angry or using strong language.",
                    ],
                ),
            },
        )
        check("official example returns typed answers", len(response.answers) == 3)
        check("grouped .nouls accessor", response.nouls["refund_requested"].noul > 0.5,
              f"noul={response.nouls['refund_requested'].noul}")
        check("grouped .choices accessor", response.choices["request_type"].choice == "refund",
              f"choice={response.choices['request_type'].choice}")
        check("choice confidence present", 0 <= response.choices["request_type"].confidence <= 1)
        check("grouped .scores accessor", 0 <= response.scores["frustration"].score <= 2,
              f"score={response.scores['frustration'].score}")
        check("response.model is versioned id", response.model == "jev-1.13.0", response.model)
        check("usage present", response.usage.input_tokens > 0,
              f"in={response.usage.input_tokens} out={response.usage.output_tokens}")

        # ---- 2. dict-style questions (docstring example) -------------------
        result = client.system_one(
            state="I was charged twice. Please help.",
            questions={
                "billing": Noul(instructions="Is this about billing?"),
                "tone": Choice(instructions="What is the tone?",
                               criteria={"calm": None, "angry": None}),
            },
        )
        check("null choice descriptions accepted", result.choices["tone"].choice in {"calm", "angry"},
              f"tone={result.choices['tone'].choice}")
        check("billing noul answered", 0 <= result.nouls["billing"].noul <= 1)

        # ---- 3. models listing (official ModelMetadataList shape) ----------
        listed = client.models.list()
        names = [m.name for m in listed.models]
        check("models listing works", "decidex-latest" in names, f"names={names}")
        check("model entries fully populated",
              all(m.description and m.release_date for m in listed.models))

        # ---- 4. error path: 422 surfaces as the official error type --------
        try:
            huge_levels = ["only", "a", *[str(i) for i in range(30)]]
            client.system_one(
                state="x",
                questions={"bad": {"type": "score", "instructions": "q", "criteria": huge_levels}},
            )
            check("invalid score levels raises 422", False, "no exception raised")
        except TypeSafeUnprocessableEntityError as err:
            check("invalid score levels raises 422", True, str(err)[:80])
        except Exception as err:  # noqa: BLE001
            check("invalid score levels raises 422", False,
                  f"wrong exception type: {type(err).__name__}: {err}")

        # ---- 5. lenient contract points from the official openapi.json -----
        relaxed = client.system_one(
            state="Something happened.",
            questions={
                "no_instructions": {"type": "noul"},  # instructions optional
            },
        )
        check("missing instructions accepted", 0 <= relaxed.nouls["no_instructions"].noul <= 1)

        structured = client.system_one(
            state="Charged twice, want money back.",
            questions={
                "rubric": Score(
                    instructions="How severe?",
                    criteria=[{"label": "low", "hint": "minor"}, {"label": "high", "hint": "major"}],
                ),
            },
        )
        check("structured score levels: legend echoes verbatim",
              # the official SDK wrapper coerces legend keys to int
              structured.scores["rubric"].legend[0] == {"label": "low", "hint": "minor"})

    print()
    if failures:
        print(f"OFFICIAL SDK COMPATIBILITY: {len(failures)} FAILURES -> {failures}")
        sys.exit(1)
    print("OFFICIAL SDK COMPATIBILITY: ALL CHECKS PASS")


if __name__ == "__main__":
    main()

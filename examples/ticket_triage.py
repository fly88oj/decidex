"""End-to-end example: customer-support ticket triage with confidence gating.

Mirrors the official "Intent routing" / "Confidence-gated routing" patterns
(docs.typesafe.ai/patterns/intent-routing, /patterns/confidence-routing):
one request fans out every question in parallel against the same state, then
plain code branches on the typed answers and their confidence.

Usage:
    python examples/ticket_triage.py [--base-url http://127.0.0.1:8600]

Start the server first:
    python -m decidex serve --engine llm --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json

from decidex import Choice, DecidexClient, Noul, Score


def triage(client: DecidexClient, ticket: dict) -> dict:
    response = client.system_one(
        state=ticket,
        questions={
            # Speculative fan-out: ask everything up front, use what matters.
            "intent": Choice(
                instructions="What is the primary intent of `ticket.message`?",
                criteria={
                    "refund": "The customer wants money returned.",
                    "bug_report": "Something is broken or erroring.",
                    "account_access": "Login, password, or access problems.",
                    "other": "Anything else.",
                },
            ),
            "refund_wanted": Noul(
                instructions="Does `ticket.message` explicitly ask for a refund or credit?"
            ),
            "urgency": Noul(
                instructions="Does `ticket.message` convey urgency or time-sensitivity?",
                criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
            ),
            "frustration": Score(
                instructions="How frustrated does the customer in `ticket.message` appear?",
                criteria=["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"],
            ),
            "severity": Score(
                instructions="How severe is the reported problem in `ticket.message`?",
                criteria=["Cosmetic; no impact", "Degraded; workaround exists", "Blocking; no workaround"],
            ),
        },
    )
    answers = response.answers

    # ---- confidence-gated routing: code decides, the model only judges ----
    action: str
    intent = answers["intent"]
    if intent.confidence < 0.35:
        action = "route_to_human"
    elif intent.choice == "refund":
        action = (
            "auto_refund_flow"
            if answers["refund_wanted"].noul > 0.6 and answers["urgency"].noul > 0.7
            else "refund_review_queue"
        )
    elif intent.choice == "bug_report":
        action = (
            "escalate_to_engineering"
            if answers["severity"].score > 1.5
            else "bug_backlog"
        )
    elif intent.choice == "account_access":
        action = "access_support_flow"
    else:
        action = "general_queue"

    return {
        "model": response.model,
        "answers": {k: vars(a) for k, a in answers.items()},
        "action": action,
    }


TICKETS = [
    {
        "ticket": {
            "id": "T-104",
            "message": "I was charged twice for order A-104 this morning. "
                       "This is unacceptable — refund the duplicate charge immediately!",
            "order": {"id": "A-104", "charges": [{"amount_usd": 49, "status": "captured"}] * 2},
        },
        "refund_policy": "Duplicate charges are eligible for a full refund.",
    },
    {
        "ticket": {
            "id": "T-105",
            "message": "The export button turns purple on Fridays but the file still downloads. "
                       "Just wanted to let you know.",
        },
    },
    {
        "ticket": {
            "id": "T-106",
            "message": "I can't log in after resetting my password. I've tried three times.",
        },
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8600")
    args = parser.parse_args()

    with DecidexClient(base_url=args.base_url) as client:
        for ticket in TICKETS:
            result = triage(client, ticket)
            print(f"== {result['model']} ==")
            print(json.dumps(result["answers"], indent=2, ensure_ascii=False))
            print(f"-> action: {result['action']}\n")


if __name__ == "__main__":
    main()

"""Use the Decidex GGUF builds through llama-server as a decision API.

The GGUFs are decision engines: they answer the lettered-option completion
prompt with a single letter. With llama-server's `n_probs` we can read the
full probability distribution over the letters at the "Answer:" position —
the same readout Decidex's native engine uses — turning any llama.cpp server
into a Jev-style decision API.

    # start the server (GPU; add --model-draft ... for MTP on generation)
    llama-server -m gguf/decidex-core-8b-r1-Q4_K_M.gguf -c 8192 --port 8080

    python examples/gguf_decision_client.py
"""

from __future__ import annotations

import argparse
import json
import math
import string
from collections import defaultdict

import httpx

TEMPLATE = """You are a precise decision engine. Evaluate the STATE against the
QUESTION and pick the single best option.

STATE:
{state}

QUESTION:
{instruction}

OPTIONS:
{options}

Answer with the letter of the single best option.
Answer:"""

LETTERS = string.ascii_uppercase


def build_prompt(state: str, instruction: str, options: list[str]) -> str:
    rendered = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, options))
    return TEMPLATE.format(state=state, instruction=instruction, options=rendered)


def letter_distribution(client: httpx.Client, base_url: str,
                        state: str, instruction: str, options: list[str]) -> list[float]:
    """Probability over the option letters at the answer position."""
    response = client.post(base_url.rstrip("/") + "/completion", json={
        "prompt": build_prompt(state, instruction, options),
        "n_predict": 1,
        "temperature": 0,
        "n_probs": 20,
        "cache_prompt": True,
    }, timeout=300)
    response.raise_for_status()
    content = response.json()

    # First generated position: aggregate the model's top-logprob mass over
    # the option letters (logprob = log p, so exp() recovers probability).
    first = content["completion_probabilities"][0]
    probs = defaultdict(float)
    for entry in first["top_logprobs"]:
        stripped = entry["token"].strip()
        if stripped in LETTERS[: len(options)]:
            probs[stripped] += math.exp(entry["logprob"])
    total = sum(probs.values())
    if total <= 0:
        raise RuntimeError("no letter tokens in top logprobs; got "
                           f"{[e['token'] for e in first['top_logprobs']]}")
    return [round(probs.get(letter, 0.0) / total, 4) for letter in LETTERS[: len(options)]]


def noul(state: str, instruction: str, client: httpx.Client, base_url: str) -> float:
    probs = letter_distribution(client, base_url, state, instruction, ["yes", "no"])
    return probs[0]


def choice(state: str, instruction: str, criteria: dict,
           client: httpx.Client, base_url: str) -> dict:
    keys = list(criteria)
    options = [f"{key} - {desc}" if desc else key for key, desc in criteria.items()]
    probs = letter_distribution(client, base_url, state, instruction, options)
    top = probs.index(max(probs))
    k = len(probs)
    confidence = max(0.0, min(1.0, (k * max(probs) - 1) / (k - 1))) if k > 1 else 1.0
    return {"choice": keys[top],
            "probabilities": dict(zip(keys, probs)),
            "confidence": round(confidence, 4)}


def score(state: str, instruction: str, levels: list[str],
          client: httpx.Client, base_url: str) -> dict:
    probs = letter_distribution(client, base_url, state, instruction, levels)
    value = sum(i * p for i, p in enumerate(probs))
    k = len(probs)
    confidence = max(0.0, min(1.0, (k * max(probs) - 1) / (k - 1))) if k > 1 else 1.0
    return {"score": round(value, 4),
            "legend": {str(i): level for i, level in enumerate(levels)},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "confidence": round(confidence, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    state = ("Help! My payouts have been failing for 3 days. "
             "I was charged twice for order A-104 and I want a refund immediately.")
    with httpx.Client() as client:
        print("noul  :", noul(state, "Does this convey urgency?", client, args.base_url))
        print("choice:", json.dumps(choice(state, "Which team should handle this?", {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        }, client, args.base_url), ensure_ascii=False))
        print("score :", json.dumps(score(state, "How frustrated is the customer?", [
            "Calm, just stating facts", "Frustrated but civil", "Very angry, strong language",
        ], client, args.base_url), ensure_ascii=False))


if __name__ == "__main__":
    main()

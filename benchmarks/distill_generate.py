"""Generate a distillation dataset from the official Jev API.

Builds diverse (state, questions) requests that share NO states or questions
with the comparison corpus (held out for evaluation), sends them to the
official API via OpenRouter, and stores each question's target distribution
over the exact option texts Decidex renders at inference time.

Output: benchmarks/distill_dataset.jsonl — one record per question:
    {"prompt": full inference prompt, "letters": ["A", "B"],
     "target": [p_A, p_B, ...], "kind": "noul|choice|score"}

    python benchmarks/distill_generate.py [--states 800] [--concurrency 4]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decidex.engines.llm_logits import LETTERS  # noqa: E402

# ---------------------------------------------------------------------------
# state generation: combinatorial templates, deliberately disjoint from the
# comparison corpus and the authored benchmark.
# ---------------------------------------------------------------------------

OPENERS = [
    "Hi team,", "Hello,", "To whom it may concern:", "Hey,", "Good morning,",
    "", "", "Quick question --", "I'm writing because", "FYI --",
]
SITUATIONS = [
    ("my subscription was renewed without notice", "billing"),
    ("the invoice for March shows twice", "billing"),
    ("my card was declined at checkout", "billing"),
    ("I never received the refund you promised", "billing"),
    ("the app freezes when I open settings", "technical"),
    ("uploads fail with error 413 every time", "technical"),
    ("the search results are empty since Monday", "technical"),
    ("the mobile app logs me out randomly", "technical"),
    ("the printed label has the wrong address", "shipping"),
    ("my parcel has been in transit for three weeks", "shipping"),
    ("the package arrived damaged", "shipping"),
    ("I want to change the delivery date", "shipping"),
    ("I forgot my password and the reset email never arrives", "account"),
    ("my account was locked after two tries", "account"),
    ("I need to update my email address", "account"),
    ("two-factor codes arrive hours late", "account"),
    ("the new dashboard is much better organized", "praise"),
    "your support agent Maria was incredibly helpful",
    "the onboarding flow was confusing at first but fine now",
    ("I was charged for the premium tier I never selected", "billing"),
    ("the export to CSV lost my column formatting", "technical"),
    ("the webinar link in the calendar invite is broken", "technical"),
]
EMOTIONS = [
    "", "", "",  # mostly neutral
    " I'm really disappointed, this keeps happening.",
    " This is the second time I report this. Seriously?",
    " Absolutely unacceptable service!!!",
    " I appreciate any help, no rush.",
    " Please fix this ASAP, we launch tomorrow!",
    " Honestly at this point I just want my money back.",
    " Best regards and thanks in advance.",
]
EXTRAS = [
    "", "", "",
    " Order number: {order}.",
    " Account: {account}.",
    " This is on the mobile app, version 4.2.",
    " It works on my colleague's machine but not mine.",
    " Screenshot attached.",
]
MULTILINGUAL = [
    "我的账户被锁定了，重置邮件一直收不到，请尽快处理。",
    "先月の請求書が二重になっていました。至急確認してください。",
    "Ich habe die Erstattung immer noch nicht erhalten.",
    "El paquete llegó dañado y quiero un reembolso.",
    "Mon abonnement a été renouvelé sans préavis.",
]
PERSONAL = [
    "You can reach me at dana.work@fastmail.com.",
    "My phone is 555-238-1147 if that helps.",
    "Shipping to 77 Birch Lane, Apt 9.",
    "Invoice ref INV-2026-00871.",
]

# ---------------------------------------------------------------------------
# question pools — mirrors the official primitive shapes; instruction wording
# is intentionally varied
# ---------------------------------------------------------------------------

NOUL_QUESTIONS = [
    "Does this message convey urgency or time-sensitivity?",
    "Is the tone of this message polite and civil?",
    "Does the writer appear frustrated?",
    "Does this message request a refund or money back?",
    "Does this message describe something broken or erroring?",
    "Does this message contain personally identifiable information?",
    "Does this message look like spam or unsolicited advertising?",
    "Is this a phishing attempt?",
    "Does the writer threaten to cancel or stop buying?",
    "Does the writer express gratitude?",
    "Is this message about an account access problem?",
    "Does this message mention a specific order or account number?",
    "Would this message normally be escalated to a human agent?",
    "Is this message primarily a complaint rather than a question?",
    "Does the writer describe a workaround for their problem?",
    "Is the writing style professional?",
    "Does the message contain sarcasm or irony?",
    "Does this look like a business rather than personal message?",
    "Is the reported issue blocking the writer's work?",
    "Does the writer ask for information rather than demand action?",
    "Does this message mention a deadline?",
    "Is the message written in English?",
    "Does the message contain hostile or aggressive language?",
    "Would a sentiment model label this message positive?",
]

SCORE_RUBRICS = [
    ("How frustrated does the writer appear?",
     ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]),
    ("How severe is the reported problem?",
     ["Cosmetic; no impact", "Degraded; workaround exists", "Blocking; no workaround"]),
    ("How complex is this request to resolve?",
     ["Simple lookup or standard procedure", "Requires judgment or multiple steps",
      "Unusual edge case, escalation needed"]),
    ("How polite is the tone of the message?",
     ["Brusque or hostile", "Neutral", "Courteous and warm"]),
    ("How urgent does this message feel?",
     ["No time pressure", "Would like attention soon", "Needs attention today",
      "Critical, immediate action required"]),
    ("How formal is the writing style?",
     ["Casual, chatty", "Everyday professional", "Formal business correspondence"]),
    ("How emotionally charged is this message?",
     ["Completely neutral", "Slightly warm or annoyed", "Clearly emotional",
      "Strongly emotional", "Extremely heated"]),
    ("How likely is this writer to churn?",
     ["No sign of leaving", "Mildly dissatisfied", "Considering alternatives",
      "Explicitly threatens to cancel"]),
    ("How much does the writer seem to trust the company?",
     ["Distrustful or hostile", "Skeptical", "Neutral", "Confident and positive"]),
]

CHOICE_SETS = [
    ("Which team should handle this?",
     {"billing": "Payments, invoicing, refunds",
      "technical": "Bugs, outages, integrations",
      "shipping": "Delivery, parcels, logistics",
      "account": "Login, access, credentials",
      "other": "Anything else"}),
    ("What is the writer's primary intent?",
     {"get_help": "Wants a problem fixed",
      "money_back": "Wants money returned",
      "information": "Just asking a question",
      "complaint": "Voicing dissatisfaction only"}),
    ("How would you classify this message?",
     {"bug_report": "Something is broken",
      "feature_request": "Asking for new functionality",
      "question": "Asking for information",
      "feedback": "Giving an opinion or praise"}),
    ("What is the emotional register of the message?",
     {"calm": "Neutral, matter-of-fact",
      "annoyed": "Mildly displeased but civil",
      "angry": "Strong displeasure or hostility",
      "cheerful": "Positive or enthusiastic"}),
]


def build_states(rng: random.Random, count: int) -> list[str]:
    states = []
    orders = [f"A-{rng.randint(10000, 99999)}" for _ in range(count)]
    accounts = [f"user{rng.randint(1000, 9999)}" for _ in range(count)]
    for i in range(count):
        if rng.random() < 0.10:
            state = rng.choice(MULTILINGUAL)
        else:
            situation = rng.choice(SITUATIONS)
            core = situation[0] if isinstance(situation, tuple) else situation
            opener = rng.choice(OPENERS)
            joiner = "" if not opener or opener.endswith(("--", ":")) else " "
            state = (
                opener + joiner + core[0].upper() + core[1:] + "."
                + rng.choice(EMOTIONS)
                + rng.choice(EXTRAS).format(order=orders[i], account=accounts[i])
            )
        if rng.random() < 0.12:
            state += " " + rng.choice(PERSONAL)
        states.append(" ".join(state.split()))
    return states


def build_questions(rng: random.Random, state: str, score_heavy: bool = False) -> dict:
    """Sample 4-8 mixed questions for a state."""
    questions = {}
    noul_picks = rng.sample(NOUL_QUESTIONS, rng.randint(2, 4) if not score_heavy else 2)
    for i, instruction in enumerate(noul_picks):
        questions[f"n{i}"] = {"type": "noul", "instructions": instruction}
    score_chance = 1.0 if score_heavy else 0.6
    if rng.random() < score_chance:
        instruction, levels = rng.choice(SCORE_RUBRICS)
        questions["s0"] = {
            "type": "score", "instructions": instruction, "criteria": list(levels),
        }
    if score_heavy or rng.random() < 0.5:
        rubrics = [r for r in SCORE_RUBRICS if r[0] != questions.get("s0", {}).get("instructions")]
        instruction, levels = rng.choice(rubrics)
        questions["s1"] = {
            "type": "score", "instructions": instruction, "criteria": list(levels),
        }
    if rng.random() < 0.4:
        instruction, criteria = rng.choice(CHOICE_SETS)
        questions["c0"] = {
            "type": "choice", "instructions": instruction, "criteria": criteria,
        }
    return questions


def option_texts(question: dict) -> list[str]:
    from decidex.render import render_text

    if question["type"] == "noul":
        criteria = question.get("criteria") or {}
        yes = f"yes - {render_text(criteria['true'])}" if criteria.get("true") else "yes"
        no = f"no - {render_text(criteria['false'])}" if criteria.get("false") else "no"
        return [yes, no]
    if question["type"] == "choice":
        out = []
        for key, description in question["criteria"].items():
            rendered = render_text(description) if description is not None else None
            out.append(f"{key} - {rendered}" if rendered is not None else str(key))
        return out
    return [render_text(level) for level in question["criteria"]]


def build_prompt(state: str, question: dict) -> str:
    from decidex.render import render_state, render_text

    engine = _PROMPT_ENGINE
    return engine.build_prompt(render_state(state), render_text(question.get("instructions")),
                               option_texts(question))


_PROMPT_ENGINE = None


def target_distribution(question: dict, answer: dict) -> list[float] | None:
    if question["type"] == "noul":
        p = answer["noul"]
        return [p, 1.0 - p]
    if question["type"] == "choice":
        keys = list(question["criteria"].keys())
        probs = answer.get("probabilities", {})
        if len(probs) != len(keys):
            return None
        return [float(probs.get(k, 0.0)) for k in keys]
    levels = question["criteria"]
    probs = answer.get("probabilities", {})
    if len(probs) != len(levels):
        return None
    return [float(probs.get(str(i), 0.0)) for i in range(len(levels))]


def fetch(client: httpx.Client, base_url: str, key: str, model: str,
          state: str, questions: dict) -> tuple[int, dict]:
    body = {"state": state, "model": model, "questions": questions}
    for attempt in range(5):
        response = client.post(base_url + "/api/alpha/decisions", json=body,
                               headers={"Authorization": f"Bearer {key}"}, timeout=180)
        if response.status_code in (429, 529) and attempt < 4:
            time.sleep(2 ** attempt)
            continue
        try:
            return response.status_code, response.json()
        except Exception:
            return response.status_code, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=800)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--score-heavy", action="store_true",
                        help="two score rubrics per state (for score-focused batches)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "distill_dataset.jsonl"))
    args = parser.parse_args()

    global _PROMPT_ENGINE
    from decidex.engines.llm_logits import LLMLogitsEngine

    class _Minimal:
        """Reuse build_prompt formatting without loading model weights."""
        build_prompt = LLMLogitsEngine.build_prompt
        build_prefix = LLMLogitsEngine.build_prefix
        build_suffix = LLMLogitsEngine.build_suffix

    _PROMPT_ENGINE = _Minimal()
    _PROMPT_ENGINE.prompt_variant = "plain"

    import os

    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip()
    rng = random.Random(args.seed)
    states = build_states(rng, args.states)
    payloads = [(state, build_questions(rng, state, args.score_heavy)) for state in states]

    print(f"states: {len(states)}, questions: {sum(len(q) for _s, q in payloads)}")

    results: list[dict] = []
    failed = 0
    with httpx.Client() as client:
        def run(item):
            state, questions = item
            return state, questions, fetch(client, "https://openrouter.ai", key,
                                           "~typesafe/jev-latest", state, questions)

        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for state, questions, (status, body) in pool.map(run, payloads):
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(payloads)} requests "
                          f"({len(results)} samples, {failed} failed)")
                if status != 200:
                    failed += 1
                    continue
                for qid, question in questions.items():
                    answer = body.get("answers", {}).get(qid)
                    if not answer:
                        continue
                    target = target_distribution(question, answer)
                    if not target or len(target) < 2:
                        continue
                    results.append({
                        "state": state, "kind": question["type"],
                        "instructions": question.get("instructions"),
                        "options": option_texts(question),
                        "letters": list(LETTERS[: len(target)]),
                        "target": target,
                    })

    rng.shuffle(results)
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    kinds = {}
    for r in results:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"saved {len(results)} samples to {args.out}: {kinds}; {failed} failed requests")


if __name__ == "__main__":
    main()

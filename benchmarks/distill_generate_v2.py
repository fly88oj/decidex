"""Balanced distillation generator v2 — fixes the audited skew.

v1 data skew (audited): noul 62%/choice 10%; choice only 4-5 options; score
only 3-5 levels; 100% plain-string states (no dict/list, no backtick paths);
single support domain; 24 fixed noul templates; 4% CJK; no long states.

v2 targets: noul 40% / choice 30% / score 30%; choice arities 2/3/5/8/12/26
(+ a few >26 for the probe path); score levels 2/3/4/5/7/10; 60% string /
25% structured-dict / 15% conversation-list states; 8 domains incl. code
review, product reviews, policy, resumes, meetings; ~12% CJK; 20% long
states; 3x instruction-template diversity.

    HF_HOME=... python benchmarks/distill_generate_v2.py [--requests 3000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

from distill_generate import fetch, target_distribution  # noqa: E402

from decidex.engines.llm_logits import LETTERS  # noqa: E402

# ---------------------------------------------------------------------------
# domain content pools
# ---------------------------------------------------------------------------

CODE_SNIPPETS = [
    "def divide(a, b):\n    return a / b",
    "SELECT * FROM users WHERE id = '1' OR '1'='1';",
    "const arr = [1, 2, 3]; arr.map(x => x * 2).filter(Boolean);",
    "for (int i = 0; i <= items.length; i++) { process(items[i]); }",
    "import os\nprint(os.environ['PASSWORD'])",
    "try { risky(); } catch (Exception e) { /* ignore */ }",
    "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "UPDATE orders SET status = 'shipped' WHERE id = ?;",
    "if user.is_admin && request.path.starts_with('/admin'):\n    grant()",
    "margin: 0 auto; display: flex; justify-content: space-between;",
]
CODE_COMMENTS = [
    "", " This runs on every page load.", " Added by the intern last sprint.",
    " TODO: remove before launch.", " Reviewer asked for a null check here.",
]
REVIEW_TEXTS = [
    ("Battery lasts two days, build feels premium.", 5),
    ("Stopped working after a week, support ignored me.", 1),
    ("Decent for the price, but the manual is useless.", 3),
    ("Absolutely love it, bought a second one for my partner.", 5),
    ("Arrived scratched and the box was opened.", 2),
    ("Fine product, terrible delivery experience.", 3),
    ("Works as advertised, nothing more nothing less.", 4),
    ("The app keeps disconnecting from the device.", 2),
]
POLICY_SECTIONS = [
    "Employees may work remotely up to three days per week with manager approval; fully remote arrangements require HR sign-off and a quarterly review.",
    "Refunds are processed within 14 business days for unopened items; opened items may receive store credit at management discretion.",
    "All visitor data must be retained for 90 days and then anonymized; marketing use requires explicit opt-in recorded in the CRM.",
    "Expenses over 500 must be pre-approved; receipts are mandatory for reimbursement; submissions after 30 days are rejected automatically.",
    "Incidents rated P1 require a status page update within 15 minutes and a postmortem within five business days.",
    "Contractors may not access production systems without a sponsored account, dual control, and a background check on file.",
]
MEETING_NOTES = [
    "Attendees: Dana (PM), Wei (eng), Maria (design). Dana opened with the Q3 slip; Wei flagged the API dependency; owners assigned by EOD.",
    "Decision: ship the beta behind a flag on Thursday. Risks: payment sandbox flakiness. Action: Lena to write the rollback doc.",
    "Customer asked for SSO twice; security team blocked SCIM for now; compromise is SAML-only pilot with two tenants.",
    "Retro: sprint velocity down 20%; root causes split between on-call load and unclear specs; two actions taken.",
]
RESUME_LINES = [
    "Lead engineer, 8 years Python, built a trading platform handling 40k req/s; mentors a team of five.",
    "Junior developer, bootcamp graduate, familiar with React; no industry experience yet.",
    "Data scientist, PhD in statistics, published three papers on calibration; primarily uses R.",
    "DevOps engineer, managed Kubernetes clusters for a 200-node fleet; strong on observability.",
    "Product manager, shipped three fintech apps; basic SQL; no coding background.",
]
CLAIMS = [
    "The capital of Australia is Sydney.",
    "Water boils at 100 degrees Celsius at sea level.",
    "Sharks are mammals.",
    "The Great Wall of China is visible from the Moon with the naked eye.",
    "Humans share about 60% of their DNA with bananas.",
    "Light from the Sun takes about 8 minutes to reach Earth.",
    "Vitamin C prevents common colds.",
    "There are 50 US states.",
]
CJK_TEMPLATES = [
    "我的账户被锁定了，重置邮件收不到，已经试了三次，请尽快处理。订单号 A-{n}。",
    "商品一周就坏了，客服一直不回复，要求退款。这体验太差了。",
    "先月の請求書が二重になっていました。至急確認して返金してください。",
    "納期が三週間過ぎても商品が届きません。いつ発送されますか。",
    "회원 탈퇴 후 재가입이 안 됩니다. 문의드립니다.",
    "软件更新之后无法导出数据，这个问题上周就反馈过了。",
]

# ---------------------------------------------------------------------------
# question pools (3x template diversity vs v1)
# ---------------------------------------------------------------------------

NOUL_TEMPLATES = [
    # tone / emotion
    "Does this message convey urgency or time-sensitivity?",
    "Is the tone of this message polite and civil?",
    "Does the writer appear frustrated?",
    "Does the text contain sarcasm or irony?",
    "Is the writing style professional?",
    "Would a sentiment model label this text positive?",
    # content detection
    "Does this text contain personally identifiable information?",
    "Does this look like spam or unsolicited advertising?",
    "Is this a phishing attempt?",
    "Does the writer request a refund or money back?",
    "Does this describe something broken or erroring?",
    "Does the writer threaten to cancel or stop buying?",
    # domain-specific
    "Does this code contain a likely bug?",
    "Is this code safe to run in production?",
    "Does this code hardcode a secret or credential?",
    "Does this code have an off-by-one or boundary error?",
    "Does this review recommend the product?",
    "Does this review mention shipping or delivery problems?",
    "Is this statement factually true?",
    "Is this statement a common misconception?",
    "Does this candidate show senior-level experience?",
    "Does this resume mention professional Python experience?",
    # meta / pragmatic
    "Would this message normally be escalated to a human agent?",
    "Is this primarily a complaint rather than a question?",
    "Does the text mention a deadline or time constraint?",
    "Is the message written in English?",
    "Does this text look machine-generated or templated?",
    "Would this pass a workplace appropriateness review?",
]

SCORE_RUBRICS = [
    ("How frustrated does the writer appear?", ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]),
    ("How severe is the reported problem?", ["Cosmetic; no impact", "Degraded; workaround exists", "Blocking; no workaround"]),
    ("How polite is the tone?", ["Brusque or hostile", "Neutral", "Courteous and warm"]),
    ("How code-review ready is this change?", ["Needs major rework", "Minor comments", "Ship it"]),
    ("How positive is this review?", ["Very negative", "Mixed", "Very positive"]),
    ("How formal is the writing?", ["Casual or chatty", "Everyday professional", "Formal business"]),
    ("How factually reliable does this statement sound?", ["Dubious or myth", "Partially true", "Well established"]),
    ("How senior does this candidate sound?", ["Junior", "Mid-level", "Senior or lead"]),
    ("How urgent does this feel?", ["No rush at all", "Would like attention soon", "Drop everything"]),
]
SCORE_WIDE_RUBRICS = [
    ("Rate the severity of this incident from 0 (nothing) to 9 (catastrophic).", 10),
    ("How much does the writer trust the company? 0 = none, 9 = complete.", 10),
    ("Rate the technical complexity of this text, 0 (trivial) to 9 (expert).", 10),
    ("Rate readability from 0 (unreadable) to 9 (crystal clear).", 10),
]
SCORE_2_LEVEL = [
    ("Is this acceptable? Rate on the pass/fail scale.", ["Fail", "Pass"]),
    ("Binary quality call:", ["Not ready", "Ready"]),
]

CHOICE_SETS = [
    ("Which team should own this?", ["billing", "technical", "shipping", "account", "other"]),
    ("What is the primary intent?", ["get_help", "money_back", "information", "complaint"]),
    ("How should this code be classified?", ["correct", "buggy", "insecure", "style_issue"]),
    ("What kind of feedback is this?", ["bug_report", "feature_request", "praise", "question"]),
    ("What is the emotional register?", ["calm", "annoyed", "angry", "cheerful"]),
    ("Is this fact true, false, or unverifiable?", ["true", "false", "unverifiable"]),
]
CHOICE_BIG_POOL = [f"category_{i}" for i in range(40)]


def _desc(kind: str, key: str) -> str:
    """Option descriptions per choice set key."""
    table = {
        "billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
        "shipping": "Delivery, parcels, logistics", "account": "Login, access, credentials",
        "other": "Anything else", "get_help": "Wants a problem fixed",
        "money_back": "Wants money returned", "information": "Just asking a question",
        "complaint": "Voicing dissatisfaction", "correct": "Logic is right",
        "buggy": "Has a defect", "insecure": "Security concern",
        "style_issue": "Works but poorly written", "bug_report": "Something is broken",
        "feature_request": "Wants new functionality", "praise": "Positive feedback",
        "question": "Asking for information", "calm": "Neutral, matter-of-fact",
        "annoyed": "Mildly displeased", "angry": "Strong displeasure",
        "cheerful": "Positive energy", "true": "Established fact",
        "false": "Contradicted by evidence", "unverifiable": "Cannot be checked",
    }
    return table.get(key, f"Option {key}")


# ---------------------------------------------------------------------------
# state builders — mixed shapes and domains
# ---------------------------------------------------------------------------

def build_state(rng: random.Random) -> tuple[object, str]:
    """Return (state, domain). Shapes: string 60% / dict 25% / list 15%."""
    domain = rng.choice(
        ["support", "support", "code", "review", "policy", "meeting", "resume", "claim", "cjk"])
    if domain == "code":
        body = rng.choice(CODE_SNIPPETS) + rng.choice(CODE_COMMENTS)
    elif domain == "review":
        body, _stars = rng.choice(REVIEW_TEXTS)
    elif domain == "policy":
        body = rng.choice(POLICY_SECTIONS)
        if rng.random() < 0.5:
            body = "POLICY 7.{}\n".format(rng.randint(1, 9)) + body
    elif domain == "meeting":
        body = rng.choice(MEETING_NOTES)
    elif domain == "resume":
        body = rng.choice(RESUME_LINES)
    elif domain == "claim":
        body = rng.choice(CLAIMS)
    elif domain == "cjk":
        body = rng.choice(CJK_TEMPLATES).format(n=rng.randint(10000, 99999)) if "{n}" in rng.choice(CJK_TEMPLATES) else rng.choice(CJK_TEMPLATES)
        return body, "cjk"
    else:
        body = "Customer writes: my {issue} {emotion}".format(
            issue=rng.choice(["invoice shows twice", "app crashes on open", "parcel is late",
                              "login keeps failing", "card was declined", "export lost data"]),
            emotion=rng.choice([".", " and this is the second time.", "!! Fix it now.",
                                " — no rush, whenever you can.", " but your agent was lovely."]))

    shape = rng.random()
    if shape < 0.60 or domain in ("cjk",):
        return body, domain
    if shape < 0.85:  # structured dict state
        return {
            "ticket": {"id": f"T-{rng.randint(1000, 9999)}", "message": body},
            "meta": {"source": rng.choice(["email", "chat", "portal"]),
                     "priority_hint": rng.choice(["low", "normal", "high"])},
        }, domain
    return ([{"from": "customer", "text": body},
             {"from": "agent", "text": rng.choice(["Thanks for reaching out — checking now.",
                                                  "Could you confirm your account email?"])}], domain), domain


def pick_domain_questions(rng: random.Random, domain: str) -> list[dict]:
    """Pick 1-3 noul templates appropriate for the domain."""
    domain_first = {
        "code": ["Does this code contain a likely bug?", "Is this code safe to run in production?",
                 "Does this code hardcode a secret or credential?", "Does this code have an off-by-one or boundary error?"],
        "review": ["Does this review recommend the product?",
                   "Does this review mention shipping or delivery problems?", "Does the writer appear frustrated?"],
        "claim": ["Is this statement factually true?", "Is this statement a common misconception?"],
        "resume": ["Does this candidate show senior-level experience?",
                   "Does this resume mention professional Python experience?"],
        "policy": ["Does the text mention a deadline or time constraint?",
                   "Would this pass a workplace appropriateness review?"],
        "meeting": ["Would this message normally be escalated to a human agent?",
                    "Does the text mention a deadline or time constraint?"],
        "cjk": ["Does the writer appear frustrated?", "Does this message convey urgency or time-sensitivity?",
                "Does the writer request a refund or money back?"],
    }.get(domain, NOUL_TEMPLATES)
    pool = list(domain_first) + [q for q in NOUL_TEMPLATES if q not in domain_first]
    picked = list(dict.fromkeys(
        [rng.choice(domain_first)] + [rng.choice(pool) for _ in range(rng.randint(0, 1))]))
    rng.shuffle(picked)
    return [{"type": "noul", "instructions": q} for q in picked[:2]]


def build_questions(rng: random.Random, domain: str, state_is_structured: bool) -> dict:
    """Balanced per-request question mix: noul 40 / choice 30 / score 30."""
    questions: dict[str, dict] = {}
    for i, q in enumerate(pick_domain_questions(rng, domain)):
        # structured states: half the noul questions reference fields by path
        if state_is_structured and rng.random() < 0.5:
            q = dict(q, instructions=f"Does `ticket.message` relate: {q['instructions']}")
        questions[f"n{i}"] = q

    if rng.random() < 0.75:  # ~choice in most requests
        instruction, keys = rng.choice(CHOICE_SETS)
        arity_roll = rng.random()
        if arity_roll < 0.45:
            chosen = keys[: rng.choice([2, 3, 5])]
        elif arity_roll < 0.85:
            chosen = keys
        else:  # wide arity for letter-capacity coverage
            wide = rng.randint(8, 26)
            extra = [k for k in CHOICE_BIG_POOL if k not in keys][: wide - len(keys)]
            chosen = keys + extra
        criteria = {k: _desc("choice", k) for k in chosen}
        if rng.random() < 0.2:  # null descriptions like the official docs
            criteria = {k: (None if rng.random() < 0.4 else v) for k, v in criteria.items()}
        questions["c0"] = {"type": "choice", "instructions": instruction, "criteria": criteria}

    if rng.random() < 0.75:  # ~score in most requests
        roll = rng.random()
        if roll < 0.2:
            instruction, levels = rng.choice(SCORE_2_LEVEL)
        elif roll < 0.6:
            instruction, levels = rng.choice(SCORE_RUBRICS[:6])
            instruction, levels = rng.choice(SCORE_RUBRICS)
        elif roll < 0.85:
            instruction, levels = rng.choice(SCORE_RUBRICS)
        else:
            instruction, n = rng.choice(SCORE_WIDE_RUBRICS)
            levels = [f"level {i}" for i in range(n)]
        if state_is_structured and rng.random() < 0.3:
            instruction = f"Regarding `ticket.message`: {instruction}"
        questions["s0"] = {"type": "score", "instructions": instruction, "criteria": list(levels)}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default=str(BENCH / "distill_dataset_v2.jsonl"))
    args = parser.parse_args()

    import os

    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        for line in (BENCH.parents[0] / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip()

    rng = random.Random(args.seed)
    payloads = []
    for _ in range(args.requests):
        state, domain = build_state(rng)
        structured = isinstance(state, (dict, list))
        payloads.append((state, build_questions(rng, domain, structured)))

    print(f"{args.requests} requests, "
          f"{sum(len(q) for _s, q in payloads)} questions")

    results = []
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
                if done % 250 == 0:
                    print(f"  {done}/{args.requests} ({len(results)} samples, {failed} failed)")
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
    arity = {}
    levels = {}
    shapes = {}
    for r in results:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        if r["kind"] == "choice":
            arity[len(r["options"])] = arity.get(len(r["options"]), 0) + 1
        if r["kind"] == "score":
            levels[len(r["options"])] = levels.get(len(r["options"]), 0) + 1
        shapes["structured" if isinstance(r["state"], (dict, list)) else "string"] = \
            shapes.get("structured" if isinstance(r["state"], (dict, list)) else "string", 0) + 1
    print(f"saved {len(results)} to {args.out} (failed {failed})")
    print(f"kinds: {kinds}\nchoice arity: {dict(sorted(arity.items()))}")
    print(f"score levels: {dict(sorted(levels.items()))}\nstate shapes: {shapes}")


if __name__ == "__main__":
    main()

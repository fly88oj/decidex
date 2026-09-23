"""Comparison corpus: identical requests for the official API and Decidex.

Every request runs against BOTH endpoints verbatim. Coverage is layered:

- official docs examples (payouts, duplicate-charge ticket with backtick
  paths, structured instructions, smart-home style states)
- the 53-item benchmark grouped into realistic multi-question fan-outs
- hard items (sarcasm, double negation, near-miss options)
- capacity edges: 10/26/100-option choices, 2/10-level scores
- structured state, unicode/multilingual, ~2k-token long state
- paired probes: duplicates (self-consistency) and 2-vs-10-question variants
  (question-count invariance), each compared per-side
- invalid requests (error-contract agreement)

Fields: id, request (official body), repeat (send it N times), pair (id of a
related request for invariance probing).
"""

from __future__ import annotations

LONG_POLICY = (
    "COMPANY SUPPORT POLICY (v4.2).\n"
    + "".join(
        f"Section {i}. Customers experiencing issues with billing, access, or "
        f"shipping should first consult the FAQ in section {max(0, i - 1)}. "
        f"Refunds for order {1000 + i} are processed within {3 + i % 5} business "
        f"days when the item is unopened. Escalation to tier {1 + i % 3} support "
        f"is authorized only after two failed resolution attempts. Agents must "
        f"log every interaction in the CRM with tag POL-{i} before closing.\n"
        for i in range(25)
    )
)

SUPPORT_TEAMS = {
    "billing": "Payments, invoicing, refunds",
    "technical": "Bugs, outages, integrations",
    "sales": "Pricing, upgrades, new accounts",
}

FRUSTRATION_LEVELS = ["Calm, just stating facts", "Frustrated but civil",
                      "Very angry, strong language"]
SEVERITY_LEVELS = ["Cosmetic; no impact", "Degraded; workaround exists",
                   "Blocking; no workaround"]


def _ticket_state(message: str, extra: dict | None = None) -> dict:
    state = {"ticket": {"id": "T-900", "message": message}}
    if extra:
        state.update(extra)
    return state


def build_corpus() -> list[dict]:
    requests: list[dict] = []

    def add(rid: str, state, questions: dict, repeat: int = 1, pair: str | None = None):
        requests.append({"id": rid, "request": {"state": state, "model": "jev-latest",
                                                "questions": questions},
                         "repeat": repeat, "pair": pair})

    # ---- official docs examples -------------------------------------------
    add("docs-payouts", "Help! My payouts have been failing for 3 days.", {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": SUPPORT_TEAMS},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    })
    add("docs-ticket-paths", {
        "ticket": {
            "subject": "Duplicate charge",
            "messages": [{"from": "customer",
                          "text": "I was charged twice for order A-104. Please refund the duplicate."}],
        },
        "order": {"id": "A-104", "charges": [{"amount_usd": 49, "status": "captured"}] * 2},
        "refund_policy": "Duplicate charges are eligible for a refund.",
    }, {
        "refund_requested": {"type": "noul",
                             "instructions": "Does `ticket.messages[0].text` request a refund?"},
        "policy_supports": {"type": "noul",
                            "instructions": "Does `refund_policy` cover this, given `order.charges`?"},
        "request_type": {"type": "choice", "instructions": "Main request in `ticket.messages[0].text`?",
                         "criteria": {"refund": "Wants money returned",
                                      "rebooking": "Wants a replacement",
                                      "information": "Asking for information only"}},
    })
    add("docs-structured-instructions", {
        "resume": {"name": "John Smith", "location": "Oakland, California",
                   "last_employer": "Google"},
        "candidate": {"name": "John Smyth", "location": "Oakland, Calif."},
    }, {
        "same_person": {"type": "noul", "instructions": {
            "record_a": "See `resume`",
            "record_b": "See `candidate`",
            "question": "Do `record_a` and `record_b` describe the same person?"}},
    })

    # ---- benchmark fan-outs (multi-question per state, official pattern) ---
    fanouts = [
        ("fan-refund", "I was charged twice for order A-104 and I want a refund immediately.",
         SUPPORT_TEAMS, FRUSTRATION_LEVELS),
        ("fan-api500", "The API returns 500 errors whenever we call /v2/users.",
         SUPPORT_TEAMS, SEVERITY_LEVELS),
        ("fan-upgrade", "We'd like to upgrade to the enterprise plan, what does it cost?",
         SUPPORT_TEAMS, FRUSTRATION_LEVELS),
        ("fan-login", "I can't log in after resetting my password. I've tried three times.",
         SUPPORT_TEAMS, FRUSTRATION_LEVELS),
    ]
    for rid, state, teams, levels in fanouts:
        add(rid, state, {
            "route": {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": teams},
            "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "polite": {"type": "noul", "instructions": "Is the tone polite and civil?"},
            "level": {"type": "score", "instructions": "Where does this fall on the scale?",
                      "criteria": levels},
        })

    # ---- hard items --------------------------------------------------------
    add("hard-sarcasm", "Oh WONDERFUL. Another crashed build. Love spending my Friday night on this.", {
        "tone": {"type": "choice", "instructions": "What is the tone of this message?",
                 "criteria": {"calm": "Neutral, just stating facts",
                              "angry": "Sarcastic or frustrated",
                              "excited": "Genuinely enthusiastic"}},
        "frustration": {"type": "score", "instructions": "How frustrated does the author appear?",
                        "criteria": FRUSTRATION_LEVELS},
    })
    add("hard-negations", {
        "messages": [
            "Honestly at this point I wouldn't say no to getting my money back.",
            "Last time another store gave me a refund, but I don't need that here - just fix the address.",
            "No rush at all - next week is fine.",
        ]
    }, {
        "wants_refund": {"type": "noul",
                         "instructions": "Does `messages[0]` accept or request a refund?"},
        "refund_here": {"type": "noul",
                        "instructions": "Does `messages[1]` request a refund for this order?"},
        "urgent": {"type": "noul", "instructions": "Does `messages[2]` convey urgency?"},
    })
    add("hard-nearmiss",
        "The export button works fine, but it would save us hours if it could also zip the files.", {
        "kind": {"type": "choice", "instructions": "What is this message?",
                 "criteria": {"bug_report": "Something is broken or erroring",
                              "feature_request": "Asking for new functionality",
                              "question": "Asking for information"}},
    })
    add("hard-pii", {
        "msg_a": "My SSN is 123-45-6789 and my card is 4111 1111 1111 1111.",
        "msg_b": "You can reach our sales team at sales@example-corp.com or 555-0100.",
    }, {
        "a_has_pii": {"type": "noul",
                     "instructions": "Does `msg_a` contain personally identifiable information?"},
        "b_has_pii": {"type": "noul",
                     "instructions": "Does `msg_b` contain personally identifiable information?"},
    })

    # ---- capacity edges ----------------------------------------------------
    add("edge-choice10", "Routing a customer email about invoices and duplicate charges.", {
        "pick": {"type": "choice", "instructions": "Which department applies?",
                 "criteria": {f"dept_{i}": f"Handles category {i}" for i in range(10)}
                 | {"billing": "Handles invoices, charges, refunds"}},
    })
    add("edge-choice26", "Classify this document: quarterly earnings press release.", {
        "pick": {"type": "choice", "instructions": "Which document type is this?",
                 "criteria": {f"type_{i}": f"Document category {i}" for i in range(25)}
                 | {"press_release": "An official earnings press release"}},
    })
    add("edge-choice100", "A customer writes: 'my card was charged twice, refund me now'.", {
        "pick": {"type": "choice", "instructions": "Pick the matching intent.",
                 "criteria": {f"intent_{i}": f"Unrelated intent number {i}" for i in range(99)}
                 | {"refund_request": "Customer wants money returned for a charge"}},
    })
    add("edge-score2", "The build takes 40 seconds instead of 35.", {
        "slow": {"type": "score", "instructions": "How slow is this?",
                 "criteria": ["Noticeable", "Painful"]},
    })
    add("edge-score10", "We lost all data after the migration and have no backup.", {
        "sev": {"type": "score", "instructions": "Rate the severity from 0 to 9.",
                "criteria": [f"Severity level {i}" for i in range(10)]},
    })
    add("edge-no-instructions", "I was charged twice for order A-104.", {
        "is_billing": {"type": "noul", "criteria": {"true": "About billing", "false": "Not about billing"}},
    })
    add("edge-null-desc", "Customer asks about shipping times to Canada.", {
        "intent": {"type": "choice", "instructions": "What is the intent?",
                   "criteria": {"question": None, "complaint": None, "other": None}},
    })

    # ---- multilingual / unicode --------------------------------------------
    add("i18n-mixed", {
        "ticket_a": "我的订单 A-104 被扣了两次款，请马上退款！",
        "ticket_b": "領収書が二重発行されています。至急対応してください。",
    }, {
        "a_refund": {"type": "noul", "instructions": "Does `ticket_a` request a refund?"},
        "b_refund": {"type": "noul", "instructions": "Does `ticket_b` report a billing problem?"},
        "a_urgent": {"type": "noul", "instructions": "Does `ticket_a` convey urgency?"},
    })

    # ---- long state (context-rot dimension) --------------------------------
    add("long-policy", LONG_POLICY, {
        "esc": {"type": "noul",
                "instructions": "Does the policy require two failed attempts before escalation?"},
        "refund_window": {"type": "noul", "instructions": "Is any refund window longer than 6 days?"},
        "section_count": {"type": "score", "instructions": "How detailed is this document?",
                          "criteria": ["A short note", "A moderate policy", "An exhaustive manual"]},
    })

    # ---- paired probes: self-consistency (repeat=2) -------------------------
    add("dup-refund", "I was charged twice for order A-104 and I want a refund immediately.", {
        "route": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": SUPPORT_TEAMS},
        "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    }, repeat=2)
    add("dup-sarcasm", "Oh WONDERFUL. Another crashed build. Love spending my Friday night on this.", {
        "tone": {"type": "choice", "instructions": "What is the tone of this message?",
                 "criteria": {"calm": "Neutral, just stating facts", "angry": "Sarcastic or frustrated",
                              "excited": "Genuinely enthusiastic"}},
    }, repeat=2)

    # ---- paired probes: question-count invariance ---------------------------
    add("inv-2q", "Never ordering again. I want my money back for this broken lamp.", {
        "intent": {"type": "choice", "instructions": "What is the customer's primary need?",
                   "criteria": {"refund": "Wants money returned", "rebooking": "Wants a replacement",
                                "information": "Asking a question"}},
    }, pair="invariance")
    add("inv-10q", "Never ordering again. I want my money back for this broken lamp.", {
        "intent": {"type": "choice", "instructions": "What is the customer's primary need?",
                   "criteria": {"refund": "Wants money returned", "rebooking": "Wants a replacement",
                                "information": "Asking a question"}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "mentions_product": {"type": "noul", "instructions": "Does this mention a specific product?"},
        "threat_churn": {"type": "noul", "instructions": "Does the customer threaten to stop buying?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": FRUSTRATION_LEVELS},
        "severity": {"type": "score", "instructions": "How severe is the reported problem?",
                     "criteria": SEVERITY_LEVELS},
        "channel": {"type": "choice", "instructions": "What channel does this look like?",
                    "criteria": {"email": "An email", "chat": "A chat message", "review": "A public review"}},
        "policy_risk": {"type": "noul", "instructions": "Does this need policy exception review?"},
    }, pair="invariance")

    # ---- single-question battery from the authored benchmark (statistical
    # power for the agreement metrics; hard items included) ------------------
    from dataset import all_items as _all_items

    for i, item in enumerate(_all_items()[:30]):
        question = {"type": item["kind"], "instructions": item["instruction"]}
        if item["criteria"]:
            question["criteria"] = item["criteria"]
        add(f"battery-{i:02d}", item["state"], {f"q{i:02d}": question})

    return requests


# Invalid requests for the error-contract comparison. Official FastAPI
# validation messages will differ in wording; we compare STATUS and the LOC
# path (which field was rejected), not the text.
def build_invalid() -> list[dict]:
    def invalid(rid: str, body):
        return {"id": rid, "request": body}

    q = {"type": "noul", "instructions": "x"}
    return [
        invalid("err-missing-state", {"questions": {"q": q}}),
        invalid("err-state-type", {"state": 42, "questions": {"q": q}}),
        invalid("err-empty-questions", {"state": "s", "questions": {}}),
        invalid("err-bad-type", {"state": "s", "questions": {"q": {"type": "essay", "instructions": "x"}}}),
        invalid("err-choice-300", {"state": "s", "questions": {"q": {
            "type": "choice", "instructions": "x",
            "criteria": {f"o{i}": None for i in range(300)}}}}),
        invalid("err-noul-bad-key", {"state": "s", "questions": {"q": {
            "type": "noul", "instructions": "x", "criteria": {"maybe": "hi"}}}}),
        invalid("err-unknown-model", {"state": "s", "model": "gpt-4o", "questions": {"q": q}}),
        invalid("err-score-nonlist", {"state": "s", "questions": {"q": {
            "type": "score", "instructions": "x", "criteria": "low,high"}}}),
    ]

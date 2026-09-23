"""Authored benchmark dataset: items with clear ground truth.

Follows the shape of SemIf's authored-decisions eval: each item is a
(state, question, options, expected) tuple where a knowledgeable person can
verify the label in seconds. Small on purpose — the bench must run in a
couple of minutes on a local GPU so every optimization can be A/B measured.
"""

from __future__ import annotations

# ---- Noul items: (state, instruction, criteria, expected_p_yes) ------------
NOUL_ITEMS: list[tuple[str, str, dict | None, int]] = [  # (state, q, criteria, expected)
    ("Help! My payouts have been failing for 3 days!",
     "Does this convey urgency or time-sensitivity?", None, 1),
    ("Thanks so much, everything works great now.", "Does this convey urgency or time-sensitivity?", None, 0),
    ("I was charged twice for order A-104, please refund the duplicate charge.",
     "Does the customer explicitly request a refund?", None, 1),
    ("I just wanted to ask what your business hours are.",
     "Does the customer explicitly request a refund?", None, 0),
    ("My SSN is 123-45-6789 and my card is 4111 1111 1111 1111.",
     "Does this message contain personally identifiable information?", None, 1),
    ("The weather looks lovely today for a walk.",
     "Does this message contain personally identifiable information?", None, 0),
    ("BUY NOW!!! 90% OFF LUXURY WATCHES, limited offer, click here!!!",
     "Is this message spam or unsolicited advertising?", None, 1),
    ("Here is the quarterly report you asked for, see attached.",
     "Is this message spam or unsolicited advertising?", None, 0),
    ("The server has been down since 2am and we are losing revenue every minute.",
     "Does this describe a production outage?", None, 1),
    ("We rotated the team schedule for next sprint.",
     "Does this describe a production outage?", None, 0),
    ("Export button turns purple on Fridays but the file still downloads fine.",
     "Does this report a blocking problem with no workaround?", None, 0),
    ("Nobody can log in since the update, support phones are on fire.",
     "Does this report a blocking problem with no workaround?", None, 1),
    ("She has 8 years of Python experience building distributed systems at scale.",
     "Does the resume mention professional distributed-systems experience?", None, 1),
    ("Enthusiastic junior developer eager to learn modern web frameworks.",
     "Does the resume mention professional distributed-systems experience?", None, 0),
    ("CONGRATULATIONS! You won a free iPhone, claim your prize now!",
     "Is this a phishing attempt?", None, 1),
    ("Reminder: your dentist appointment is tomorrow at 10am.",
     "Is this a phishing attempt?", None, 0),
    # with criteria
    ("ASAP please, need this fixed today!", "Does this convey urgency?",
     {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}, 1),
    ("Whenever you get a chance, no rush at all.", "Does this convey urgency?",
     {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}, 0),
]

# ---- Choice items: (state, instruction, criteria, expected_key) ------------
CHOICE_ITEMS: list[tuple[str, str, dict, str]] = [
    ("I was charged twice for order A-104.", "Which team should handle this?",
     {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
      "sales": "Pricing, upgrades, new accounts"}, "billing"),
    ("The API returns 500 errors whenever we call /v2/users.", "Which team should handle this?",
     {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
      "sales": "Pricing, upgrades, new accounts"}, "technical"),
    ("We'd like to upgrade to the enterprise plan, what does it cost?", "Which team should handle this?",
     {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
      "sales": "Pricing, upgrades, new accounts"}, "sales"),
    ("Le motoeur fait un bruit étrange quand je freine.", "What language is this text written in?",
     {"english": "Text written in English", "french": "Text written in French",
      "german": "Text written in German", "spanish": "Text written in Spanish"}, "french"),
    ("Ich habe mein Passwort vergessen und kann mich nicht einloggen.",
     "What language is this text written in?",
     {"english": "Text written in English", "french": "Text written in French",
      "german": "Text written in German", "spanish": "Text written in Spanish"}, "german"),
    ("The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
     "What language is this text written in?",
     {"english": "Text written in English", "french": "Text written in French",
      "german": "Text written in German", "spanish": "Text written in Spanish"}, "english"),
    ("def add(a, b):\n    return a + b", "What programming language is this snippet?",
     {"python": "Python source code", "javascript": "JavaScript source code",
      "rust": "Rust source code", "sql": "SQL query"}, "python"),
    ("SELECT COUNT(*) FROM orders WHERE created_at > '2026-01-01';",
     "What programming language is this snippet?",
     {"python": "Python source code", "javascript": "JavaScript source code",
      "rust": "Rust source code", "sql": "SQL query"}, "sql"),
    ("fn main() { println!(\"hello\"); }", "What programming language is this snippet?",
     {"python": "Python source code", "javascript": "JavaScript source code",
      "rust": "Rust source code", "sql": "SQL query"}, "rust"),
    ("My flight was cancelled and I need to be in Zurich tomorrow morning.",
     "What is the customer's primary need?",
     {"rebooking": "Wants a replacement flight", "refund": "Wants money returned",
      "information": "Just asking a question"}, "rebooking"),
    ("Never ordering again. I want my money back for this broken lamp.",
     "What is the customer's primary need?",
     {"rebooking": "Wants a replacement flight", "refund": "Wants money returned",
      "information": "Just asking a question"}, "refund"),
    ("Do you ship to Canada and how long does it take?",
     "What is the customer's primary need?",
     {"rebooking": "Wants a replacement flight", "refund": "Wants money returned",
      "information": "Just asking a question"}, "information"),
]

# ---- Score items: (state, instruction, levels, expected_level) -------------
SCORE_ITEMS: list[tuple[str, str, list[str], int]] = [
    ("Thanks, just checking in — no rush.", "How frustrated does the customer appear?",
     ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"], 0),
    ("This is the third time I'm asking about this. Getting really tired of it.",
     "How frustrated does the customer appear?",
     ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"], 1),
    ("This is ABSOLUTELY UNACCEPTABLE!!! I have called THREE TIMES and nobody fixes ANYTHING!!!",
     "How frustrated does the customer appear?",
     ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"], 2),
    ("The logo is slightly off-center on the about page.", "How severe is the reported issue?",
     ["Cosmetic; no impact", "Degraded; workaround exists", "Blocking; no workaround"], 0),
    ("Search is slow and sometimes times out, but retrying usually works.",
     "How severe is the reported issue?",
     ["Cosmetic; no impact", "Degraded; workaround exists", "Blocking; no workaround"], 1),
    ("All payments fail with an error since this morning; we cannot sell anything.",
     "How severe is the reported issue?",
     ["Cosmetic; no impact", "Degraded; workaround exists", "Blocking; no workaround"], 2),
    ("How do I reset my password?", "How complex is this support request to resolve?",
     ["Simple lookup or standard procedure", "Requires judgment or multiple steps",
      "Unusual edge case, escalation needed"], 0),
    ("We merged two accounts by mistake and now billing history is duplicated across regions.",
     "How complex is this support request to resolve?",
     ["Simple lookup or standard procedure", "Requires judgment or multiple steps",
      "Unusual edge case, escalation needed"], 2),
    ("He used Python at his last job for two years.", "Depth of Python experience shown?",
     ["None mentioned", "Mentioned or used casually", "Primary language, deep expertise"], 1),
    ("She architected Python trading systems handling 50k req/s and mentors the platform team.",
     "Depth of Python experience shown?",
     ["None mentioned", "Mentioned or used casually", "Primary language, deep expertise"], 2),
]


# ---- Hard items: sarcasm, negation traps, near-miss options. These give the
# calibration metrics real errors to work with (the easy set saturates).
HARD_NOUL_ITEMS: list[tuple[str, str, dict | None, int]] = [
    # double negative => wants the refund
    ("Honestly at this point I wouldn't say no to getting my money back for this.",
     "Does the customer accept or want a refund?", None, 1),
    # polite but firm request buried in pleasantries
    ("Hope you're well! Quick note: the duplicate charge from Tuesday still shows on my "
     "statement. I'd appreciate it being reversed. Thanks so much!",
     "Does the customer explicitly request a refund?", None, 1),
    # refund mentioned but NOT requested
    ("Last time this happened, another store gave me a refund, but I don't need that here - "
     "just fix the shipping address please.",
     "Does the customer explicitly request a refund?", None, 0),
    # urgency words present but explicitly negated
    ("No rush at all - whenever you have a moment next week is fine.",
     "Does this convey urgency or time-sensitivity?", None, 0),
    # business contact info (not personal PII)
    ("You can reach our sales team at sales@example-corp.com or 555-0100.",
     "Does this message contain personally identifiable information?", None, 0),
    ("My home address is 42 Elm Street and my phone is 555-987-6543.",
     "Does this message contain personally identifiable information?", None, 1),
]

HARD_CHOICE_ITEMS: list[tuple[str, str, dict, str]] = [
    # sarcasm: angry words, literally positive
    ("Oh WONDERFUL. Another crashed build. Love spending my Friday night on this.",
     "What is the tone of this message?",
     {"calm": "Neutral, just stating facts", "angry": "Sarcastic or frustrated",
      "excited": "Genuinely enthusiastic"}, "angry"),
    # near-miss: feature request phrased as a bug
    ("The export button works fine, but it would save us hours if it could also zip the files.",
     "What is this message?",
     {"bug_report": "Something is broken or erroring",
      "feature_request": "Asking for new functionality",
      "question": "Asking for information"}, "feature_request"),
    # question-phrased bug report (question mark, but something is broken)
    ("Is the login button supposed to spin forever? Mine has been spinning "
     "for ten minutes and never logs me in.",
     "What is this message?",
     {"bug_report": "Something is broken or erroring",
      "feature_request": "Asking for new functionality",
      "question": "Asking for information"}, "bug_report"),
    # invoice question is billing even though it mentions the website
    ("Your website says invoices are downloadable, but I can't find where. Do you also mail them?",
     "Which team should handle this?",
     {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations",
      "sales": "Pricing, upgrades, new accounts"}, "billing"),
]

HARD_SCORE_ITEMS: list[tuple[str, str, list[str], int]] = [
    # sarcasm reads as frustrated, not calm
    ("Oh GREAT, another billing error. Just what I always wanted. Fantastic work.",
     "How frustrated does the customer appear?",
     ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"], 2),
    # issue resolved, mild residual annoyance
    ("It's sorted now, thanks - though it did take three emails to get here.",
     "How frustrated does the customer appear?",
     ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"], 1),
    # blocking for a hobby project (severity judged by impact, not emotion)
    ("My personal blog's comment section is completely broken; visitors can't post at all.",
     "How severe is the reported issue?",
     ["Cosmetic; no impact", "Degraded; workaround exists", "Blocking; no workaround"], 2),
]


def all_items() -> list[dict]:
    """Flatten the dataset into evaluation jobs with expected outcomes."""
    items = []
    for state, instruction, criteria, expected in NOUL_ITEMS + HARD_NOUL_ITEMS:
        items.append({"kind": "noul", "state": state, "instruction": instruction,
                      "criteria": criteria, "expected": expected})
    for state, instruction, criteria, expected in CHOICE_ITEMS + HARD_CHOICE_ITEMS:
        items.append({"kind": "choice", "state": state, "instruction": instruction,
                      "criteria": criteria, "expected": expected})
    for state, instruction, levels, expected in SCORE_ITEMS + HARD_SCORE_ITEMS:
        items.append({"kind": "score", "state": state, "instruction": instruction,
                      "criteria": levels, "expected": expected})
    return items

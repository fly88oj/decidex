"""SDK tests: typed questions round-trip to typed answers over real HTTP.

The SDK is exercised against a live uvicorn server on a random local port, so
the full request path (middleware, auth, serialization) is covered.
"""

import socket
import threading
import time

import pytest

from decidex import Choice, DecidexClient, DecidexError, Noul, Score
from decidex.engines import StubEngine
from decidex.server import create_app


@pytest.fixture()
def base_url():
    import uvicorn

    app = create_app(engine=StubEngine())
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while time.time() < deadline and not server.started:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("test server failed to start")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture()
def sdk(base_url):
    with DecidexClient(base_url=base_url) as client:
        yield client


STATE = {
    "ticket": {
        "subject": "Duplicate charge",
        "messages": [
            {"from": "customer",
             "text": "I was charged twice for order A-104. Please refund the duplicate."},
        ],
    },
    "refund_policy": "Duplicate charges are eligible for a refund.",
}

QUESTIONS = {
    "refund_requested": Noul(instructions="Does `ticket.messages[0].text` request a refund?"),
    "request_type": Choice(
        instructions="What is the main request in `ticket.messages[0].text`?",
        criteria={
            "refund": "The customer wants money returned.",
            "rebooking": "The customer wants a replacement.",
            "information": "The customer is asking for information only.",
        },
    ),
    "frustration": Score(
        instructions="How frustrated does the customer appear in `ticket.messages[0].text`?",
        criteria=["Calm and neutral.", "Concerned but civil.", "Very angry or using strong language."],
    ),
}


class TestQuestionBuilders:
    def test_noul_payload(self):
        assert Noul(instructions="q").to_payload() == {"type": "noul", "instructions": "q"}

    def test_noul_payload_with_criteria(self):
        payload = Noul(instructions="q", criteria={"true": "yes-ish"}).to_payload()
        assert payload["criteria"] == {"true": "yes-ish"}

    def test_choice_and_score_payloads(self):
        assert Choice(instructions="q", criteria={"a": "A"}).to_payload() == {
            "type": "choice", "instructions": "q", "criteria": {"a": "A"},
        }
        assert Score(instructions="q", criteria=["lo", "hi"]).to_payload() == {
            "type": "score", "instructions": "q", "criteria": ["lo", "hi"],
        }


class TestClientRoundTrip:
    def test_system_one_typed_answers(self, sdk):
        response = sdk.system_one(state=STATE, questions=QUESTIONS)
        assert response.model
        assert response.usage.input_tokens > 0

        noul = response.answers["refund_requested"]
        assert isinstance(noul.noul, float) and 0.0 <= noul.noul <= 1.0

        choice = response.answers["request_type"]
        assert choice.choice in {"refund", "rebooking", "information"}
        assert sum(choice.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert choice.choice == max(choice.probabilities, key=choice.probabilities.get)
        assert 0.0 <= choice.confidence <= 1.0

        score = response.answers["frustration"]
        assert 0.0 <= score.score <= 2.0
        assert score.legend["0"] == "Calm and neutral."
        assert sum(score.probabilities.values()) == pytest.approx(1.0, abs=1e-6)

    def test_dict_questions_accepted(self, sdk):
        response = sdk.system_one(
            state="plain text state",
            questions={"q": {"type": "noul", "instructions": "is it true?"}},
        )
        assert response.answers["q"].noul is not None

    def test_models_endpoint(self, sdk):
        names = [m["name"] for m in sdk.models()]
        assert "decidex-latest" in names

    def test_error_surfaces_field(self, sdk):
        with pytest.raises(DecidexError) as excinfo:
            sdk.system_one(state=42, questions={"q": Noul(instructions="x")})
        assert excinfo.value.status_code == 422
        assert excinfo.value.field == "state"

    def test_context_manager(self, base_url):
        with DecidexClient(base_url=base_url) as sdk:
            assert sdk.system_one(state="s", questions={"q": Noul(instructions="x")}).answers

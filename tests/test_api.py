"""API contract tests against the official schema from docs.typesafe.ai/api."""

import pytest
from fastapi.testclient import TestClient

from decidex import MODEL_ALIAS, MODEL_ID
from decidex.engines import StubEngine
from decidex.server import create_app


@pytest.fixture()
def client():
    app = create_app(engine=StubEngine())
    with TestClient(app) as test_client:
        yield test_client


def _detail(response) -> list:
    return response.json()["detail"]


def _loc(response) -> list:
    return _detail(response)[0]["loc"]


def _msg(response) -> str:
    return _detail(response)[0]["msg"]


VALID_REQUEST = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "decidex-latest",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


class TestHappyPath:
    def test_full_request_shape(self, client):
        response = client.post("/v1/systemone", json=VALID_REQUEST)
        assert response.status_code == 200
        body = response.json()

        assert set(body) == {"model", "answers", "usage"}
        assert body["model"] == MODEL_ID
        assert set(body["answers"]) == set(VALID_REQUEST["questions"])
        assert set(body["usage"]) == {"input_tokens", "output_tokens"}
        assert body["usage"]["input_tokens"] > 0
        # 3 choice options + 3 score levels + 1 noul decision
        assert body["usage"]["output_tokens"] == 7

    def test_noul_answer_shape(self, client):
        body = client.post("/v1/systemone", json=VALID_REQUEST).json()
        noul = body["answers"]["is_urgent"]
        assert noul["type"] == "noul"
        assert isinstance(noul["noul"], float)
        assert 0.0 <= noul["noul"] <= 1.0
        assert "confidence" not in noul  # nouls carry belief itself, no confidence
        assert "probabilities" not in noul

    def test_choice_answer_shape(self, client):
        body = client.post("/v1/systemone", json=VALID_REQUEST).json()
        choice = body["answers"]["department"]
        assert choice["type"] == "choice"
        assert choice["choice"] in VALID_REQUEST["questions"]["department"]["criteria"]
        probs = choice["probabilities"]
        assert set(probs) == set(VALID_REQUEST["questions"]["department"]["criteria"])
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
        assert choice["choice"] == max(probs, key=probs.get)
        assert 0.0 <= choice["confidence"] <= 1.0

    def test_score_answer_shape(self, client):
        body = client.post("/v1/systemone", json=VALID_REQUEST).json()
        score = body["answers"]["frustration"]
        criteria = VALID_REQUEST["questions"]["frustration"]["criteria"]
        assert score["type"] == "score"
        assert score["legend"] == {str(i): level for i, level in enumerate(criteria)}
        assert set(score["probabilities"]) == {"0", "1", "2"}
        assert sum(score["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
        expected = sum(int(i) * p for i, p in score["probabilities"].items())
        assert score["score"] == pytest.approx(expected, abs=1e-3)
        assert 0.0 <= score["confidence"] <= 1.0

    def test_structured_state_and_instructions(self, client):
        request = {
            "state": {"ticket": {"messages": [{"text": "Refund me now!"}]}},
            "questions": {
                "wants_refund": {
                    "type": "noul",
                    "instructions": {
                        "question": "Does `ticket.messages[0].text` request a refund?",
                        "policy": "Refunds allowed within 30 days.",
                    },
                },
            },
        }
        body = client.post("/v1/systemone", json=request).json()
        assert body["answers"]["wants_refund"]["type"] == "noul"

    def test_noul_with_criteria(self, client):
        request = {
            "state": "ASAP please",
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
                },
            },
        }
        body = client.post("/v1/systemone", json=request).json()
        assert 0.0 <= body["answers"]["urgent"]["noul"] <= 1.0

    def test_choice_null_description(self, client):
        request = {
            "state": "something",
            "questions": {
                "pick": {
                    "type": "choice",
                    "instructions": "Pick one",
                    "criteria": {"a": "The first thing", "b": None},
                },
            },
        }
        body = client.post("/v1/systemone", json=request).json()
        assert body["answers"]["pick"]["probabilities"]["b"] >= 0.0

    def test_model_alias_resolution(self, client):
        request = dict(VALID_REQUEST, model="decidex-1.0.0")
        assert client.post("/v1/systemone", json=request).json()["model"] == "decidex-1.0.0"
        request = dict(VALID_REQUEST, model="jev-latest")
        assert client.post("/v1/systemone", json=request).json()["model"] == "jev-1.13.0"

    def test_255_options_allowed(self, client):
        criteria = {f"opt_{i}": f"Description {i}" for i in range(255)}
        request = {"state": "s", "questions": {
            "big": {"type": "choice", "instructions": "Pick", "criteria": criteria}}}
        response = client.post("/v1/systemone", json=request)
        assert response.status_code == 200
        probs = response.json()["answers"]["big"]["probabilities"]
        assert len(probs) == 255
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


class TestValidationErrors:
    def post(self, client, body):
        return client.post("/v1/systemone", json=body)

    def test_missing_state(self, client):
        response = self.post(client, {"questions": {"q": {"type": "noul", "instructions": "x"}}})
        assert response.status_code == 422
        assert _loc(response) == ["body", "state"]

    def test_bad_state_type(self, client):
        response = self.post(client, {"state": 42, "questions": {"q": {"type": "noul", "instructions": "x"}}})
        assert response.status_code == 422
        assert _loc(response) == ["body", "state"]

    def test_empty_questions(self, client):
        response = self.post(client, {"state": "s", "questions": {}})
        assert response.status_code == 422
        assert _loc(response) == ["body", "questions"]

    def test_bad_question_type(self, client):
        request = {"state": "s", "questions": {"q": {"type": "essay", "instructions": "x"}}}
        response = self.post(client, request)
        assert response.status_code == 422
        assert _loc(response) == ["body", "questions", "q", "type"]

    def test_missing_instructions_accepted(self, client):
        # The official openapi.json marks instructions optional on all three
        # question types; a missing instruction renders as empty text.
        request = {"state": "I was charged twice.", "questions": {"q": {"type": "noul"}}}
        response = self.post(client, request)
        assert response.status_code == 200
        assert 0.0 <= response.json()["answers"]["q"]["noul"] <= 1.0

    def test_noul_null_criteria_values_accepted(self, client):
        # Official NoulCriteria allows null for true/false.
        request = {"state": "s", "questions": {
            "q": {"type": "noul", "instructions": "x", "criteria": {"true": None, "false": None}}}}
        assert self.post(client, request).status_code == 200

    def test_choice_too_many_options(self, client):
        criteria = {f"o{i}": None for i in range(256)}
        request = {"state": "s", "questions": {
            "q": {"type": "choice", "instructions": "x", "criteria": criteria}}}
        response = self.post(client, request)
        assert response.status_code == 422
        assert "255" in _msg(response)

    def test_choice_empty_criteria(self, client):
        request = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}}
        assert self.post(client, request).status_code == 422

    def test_score_one_level_accepted(self, client):
        # The official openapi.json sets min_length=1 on score criteria
        # (docs prose recommends 2+, but the server accepts 1).
        request = {"state": "s", "questions": {
            "q": {"type": "score", "instructions": "x", "criteria": ["only"]}}}
        response = self.post(client, request)
        assert response.status_code == 200
        answer = response.json()["answers"]["q"]
        assert answer["score"] == 0.0
        assert answer["confidence"] == 1.0

    def test_score_ten_levels_accepted(self, client):
        levels = [str(i) for i in range(10)]
        request = {"state": "s", "questions": {
            "q": {"type": "score", "instructions": "x", "criteria": levels}}}
        assert self.post(client, request).status_code == 200

    def test_score_too_many_levels(self, client):
        # 26 letters is the letter-readout capacity; official docs prose caps
        # at 10, we accept up to 26 and reject beyond with a clear error.
        levels = [str(i) for i in range(27)]
        request = {"state": "s", "questions": {
            "q": {"type": "score", "instructions": "x", "criteria": levels}}}
        response = self.post(client, request)
        assert response.status_code == 422
        assert _loc(response) == ["body", "questions", "q", "criteria"]

    def test_noul_bad_criteria_key(self, client):
        request = {"state": "s", "questions": {
            "q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "hi"}}}}
        response = self.post(client, request)
        assert response.status_code == 422
        assert _loc(response) == ["body", "questions", "q", "criteria"]

    def test_unknown_model(self, client):
        request = dict(VALID_REQUEST, model="gpt-4o")
        response = self.post(client, request)
        assert response.status_code == 422
        assert _loc(response) == ["body", "model"]

    def test_invalid_json_body(self, client):
        response = client.post("/v1/systemone", content=b"{not json",
                                headers={"Content-Type": "application/json"})
        assert response.status_code == 422


class TestOtherEndpoints:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["engine"] == "stub"

    def test_models_official_shape(self, client):
        # Official ModelMetadataList: models with name/description/release_date.
        body = client.get("/v1/models").json()
        names = [m["name"] for m in body["models"]]
        assert MODEL_ALIAS in names and MODEL_ID in names
        for entry in body["models"]:
            assert set(entry) == {"name", "description", "release_date"}
            assert entry["description"] and entry["release_date"]


class TestAuth:
    def test_401_without_key(self):
        app = create_app(engine=StubEngine(), api_key="secret")
        with TestClient(app) as client:
            response = client.post("/v1/systemone", json=VALID_REQUEST)
            assert response.status_code == 401
            assert "detail" in response.json()

    def test_401_wrong_key(self):
        app = create_app(engine=StubEngine(), api_key="secret")
        with TestClient(app) as client:
            response = client.post(
                "/v1/systemone", json=VALID_REQUEST, headers={"Authorization": "Bearer wrong"}
            )
            assert response.status_code == 401

    def test_bearer_ok(self):
        app = create_app(engine=StubEngine(), api_key="secret")
        with TestClient(app) as client:
            response = client.post(
                "/v1/systemone", json=VALID_REQUEST, headers={"Authorization": "Bearer secret"}
            )
            assert response.status_code == 200

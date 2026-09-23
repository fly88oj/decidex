"""Engine smoke tests. The stub engine always runs; ML engines run when their
dependencies (and cached models) are available and are marked slow."""

import importlib.util

import pytest

from decidex.calib import weighted_score
from decidex.engines import StubEngine, build_engine

HAVE_ST = importlib.util.find_spec("sentence_transformers") is not None
HAVE_HF = (importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None)

STATE = "I was charged twice for order A-104 and I want a refund immediately."
CHOICE_OPTIONS = [
    "billing - Payments, invoicing, refunds",
    "technical - Bugs, outages, integrations",
    "sales - Pricing, upgrades, new accounts",
]
SCORE_LEVELS = ["Calm and neutral", "Frustrated but civil", "Very angry, strong language"]
INSTRUCTION = "Which team should handle this support ticket?"


def _check_distribution(probs, n):
    assert len(probs) == n
    assert all(p > 0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)


class TestStubEngine:
    def test_stub_choice_prefers_overlapping_option(self):
        engine = StubEngine()
        probs = engine.score(STATE, INSTRUCTION, CHOICE_OPTIONS)
        _check_distribution(probs, 3)
        assert probs[0] == max(probs)  # "charged/refund" overlap with billing

    def test_stub_batch_independent(self):
        engine = StubEngine()
        a = engine.score(STATE, INSTRUCTION, CHOICE_OPTIONS)
        b = engine.score("something unrelated about kites", INSTRUCTION, CHOICE_OPTIONS)
        assert a != b


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_ST, reason="sentence-transformers not installed")
class TestEmbeddingEngine:
    def test_semantic_choice(self):
        engine = build_engine("embedding")
        probs = engine.score(STATE, INSTRUCTION, CHOICE_OPTIONS)
        _check_distribution(probs, 3)
        assert probs[0] == max(probs), "refund/charge state should rank billing first"

    def test_full_pipeline(self):
        from decidex.server import evaluate

        engine = build_engine("embedding")
        answers = evaluate(
            engine,
            STATE,
            {
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": SCORE_LEVELS,
                }
            },
        )
        probs = answers["frustration"]["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
        expected = weighted_score(list(probs.values()))
        assert answers["frustration"]["score"] == pytest.approx(expected, abs=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_HF, reason="torch/transformers not installed")
class TestLLMEngine:
    def test_direct_letter_readout(self):
        engine = build_engine("llm")  # default Qwen/Qwen3-4B
        probs = engine.score(STATE, INSTRUCTION, CHOICE_OPTIONS)
        _check_distribution(probs, 3)
        assert probs[0] == max(probs), "refund state should rank billing first"

    def test_noul_two_way(self):
        engine = build_engine("llm")
        urgent = engine.score("Help! My payouts have been failing for 3 days!",
                              "Does this convey urgency?", ["yes", "no"])
        calm = engine.score("Thanks, everything works great.", "Does this convey urgency?", ["yes", "no"])
        _check_distribution(urgent, 2)
        _check_distribution(calm, 2)
        assert urgent[0] > calm[0], "urgent state should score higher on yes"

    def test_high_cardinality_relevance_path(self):
        engine = build_engine("llm")
        options = [f"option_{i} - unrelated filler number {i}" for i in range(30)]
        options[7] = "refunds - customer wants money returned for a duplicate charge"
        probs = engine.score(STATE, "Which option applies?", options)
        _check_distribution(probs, 30)
        assert probs[7] == max(probs)

    def test_mixed_batch_results_stay_aligned(self):
        # Regression: a >26-option job expands into relevance probes while
        # small jobs use direct readout; results must stay in job order.
        engine = build_engine("llm")
        big_options = [f"topic_{i} - filler description {i}" for i in range(30)]
        jobs = [
            (STATE, "Which team should handle this?", CHOICE_OPTIONS),          # direct
            (STATE, "Which option applies?", big_options),                      # probe path
            (STATE, "Does this convey urgency?", ["yes", "no"]),                # direct
        ]
        results = engine.score_batch(jobs)
        assert len(results) == 3
        _check_distribution(results[0], 3)
        _check_distribution(results[1], 30)
        _check_distribution(results[2], 2)
        assert results[0][0] == max(results[0])  # billing wins on the refund state
        assert results[2][0] > results[2][1]     # refund state reads as urgent

    def test_prefix_reuse_matches_full_prompts(self):
        # The KV-prefix-reuse path must agree with full-prompt scoring.
        engine = build_engine("llm")
        assert engine.prefix_reuse, "prefix reuse should be on by default"
        jobs = [
            (STATE, "Which team should handle this?", CHOICE_OPTIONS),
            (STATE, "Does this convey urgency?", ["yes", "no"]),
            (STATE, "How frustrated is the customer?", SCORE_LEVELS),
        ]
        reused = engine.score_batch(jobs)   # shared state -> prefix path
        engine.prefix_reuse = False
        full = engine.score_batch(jobs)     # same jobs, full prompts
        engine.prefix_reuse = True
        for i, (a, b) in enumerate(zip(reused, full)):
            assert a.index(max(a)) == b.index(max(b)), f"argmax differs at job {i}"
            assert max(abs(x - y) for x, y in zip(a, b)) < 0.02, f"probs differ at job {i}"

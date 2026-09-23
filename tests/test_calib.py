"""Tests for the probability math, checked against the official API examples."""

import pytest

from decidex.calib import confidence, round_distribution, softmax, weighted_score


class TestConfidence:
    def test_official_choice_example(self):
        # docs.typesafe.ai/api choice example: p_max=0.88, K=3 -> 0.81
        probs = [0.88, 0.12, 0.0]
        assert confidence(probs) == pytest.approx(0.82, abs=0.011)

    def test_official_score_example(self):
        # docs score example: p_max=0.95, K=3 -> 0.92
        probs = [0.0, 0.95, 0.05]
        assert confidence(probs) == pytest.approx(0.925, abs=0.006)

    def test_official_formula_from_confidence_page(self):
        # The docs' interactive demo computes (3*peak - 1)/2 for K=3.
        for peak in (0.4, 0.6, 0.9, 1.0):
            rest = (1 - peak) / 2
            assert confidence([peak, rest, rest]) == pytest.approx((3 * peak - 1) / 2)

    def test_uniform_is_zero(self):
        assert confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
        assert confidence([0.5, 0.5]) == pytest.approx(0.0)

    def test_degenerate_is_one(self):
        assert confidence([1.0, 0.0, 0.0]) == 1.0
        assert confidence([1.0]) == 1.0

    def test_never_below_zero(self):
        # Mathematically p_max >= 1/K for any distribution summing to 1, so
        # confidence >= 0 always; near-uniform inputs must land near 0.
        assert confidence([0.333, 0.333, 0.334]) == pytest.approx(0.001, abs=1e-9)
        assert confidence([0.5, 0.5]) == pytest.approx(0.0)


class TestWeightedScore:
    def test_official_example(self):
        # {0: 0.0, 1: 0.95, 2: 0.05} -> score 1.05 (exact per docs)
        assert weighted_score([0.0, 0.95, 0.05]) == pytest.approx(1.05)

    def test_can_land_between_levels(self):
        assert weighted_score([0.5, 0.5]) == pytest.approx(0.5)


class TestSoftmax:
    def test_sums_to_one(self):
        probs = softmax([2.0, -1.0, 0.3, 7.7])
        assert sum(probs) == pytest.approx(1.0)
        assert all(p > 0 for p in probs)

    def test_temperature_sharpens(self):
        logits = [3.0, 1.0]
        hot = softmax(logits, temperature=0.1)
        cold = softmax(logits, temperature=10.0)
        assert max(hot) > max(cold)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            softmax([1.0], temperature=0.0)


class TestRoundDistribution:
    def test_sums_to_exactly_one_after_rounding(self):
        probs = softmax([0.1, 0.25, -0.4, 3.2])
        rounded = round_distribution(probs)
        assert sum(rounded) == pytest.approx(1.0, abs=1e-9)

    def test_argmax_survives(self):
        probs = softmax([5.0, 0.0, 0.0])
        rounded = round_distribution(probs)
        assert rounded.index(max(rounded)) == 0

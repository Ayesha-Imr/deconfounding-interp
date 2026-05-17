"""Tests for logprobs-based judge scoring."""

import math

from deconfounding_interp.llm.judge import get_judge_score_from_logprobs


def _make_logprobs(token_logprob_pairs):
    """Helper: build logprobs structure from [(token, logprob), ...]."""
    return [{"top_logprobs": [{"token": t, "logprob": lp} for t, lp in token_logprob_pairs]}]


def test_single_dominant_token():
    logprobs = _make_logprobs([("0", -0.0001), ("5", -9.4), ("2", -12.9)])
    score = get_judge_score_from_logprobs(logprobs)
    assert score is not None
    assert score < 0.01  # "0" dominates


def test_mixed_tokens():
    logprobs = _make_logprobs([("50", math.log(0.5)), ("80", math.log(0.3)), ("20", math.log(0.2))])
    score = get_judge_score_from_logprobs(logprobs)
    expected = (50 * 0.5 + 80 * 0.3 + 20 * 0.2) / (0.5 + 0.3 + 0.2)
    assert score is not None
    assert abs(score - expected) < 0.01


def test_out_of_range_tokens_ignored():
    logprobs = _make_logprobs([("50", -0.01), ("150", -0.5), ("-1", -0.5)])
    score = get_judge_score_from_logprobs(logprobs)
    assert score is not None
    assert abs(score - 50.0) < 1.0  # only "50" is valid


def test_non_integer_tokens_ignored():
    logprobs = _make_logprobs([("75", -0.01), ("high", -0.5), ("yes", -0.5)])
    score = get_judge_score_from_logprobs(logprobs)
    assert score is not None
    assert abs(score - 75.0) < 1.0


def test_low_probability_returns_none():
    logprobs = _make_logprobs([("hello", -0.01), ("world", -0.5)])
    score = get_judge_score_from_logprobs(logprobs)
    assert score is None


def test_empty_logprobs():
    assert get_judge_score_from_logprobs([]) is None
    assert get_judge_score_from_logprobs([{"top_logprobs": []}]) is None


def test_custom_scale():
    logprobs = _make_logprobs([("5", -0.01), ("8", -2.0)])
    score = get_judge_score_from_logprobs(logprobs, scale_min=0, scale_max=10)
    assert score is not None
    assert 4.5 < score < 5.5  # "5" dominates

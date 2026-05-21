"""Tests for downstream evaluation aggregation logic."""


from deconfounding_interp.pipelines.downstream_evaluation import _compute_aggregates


def _resp(alpha, trait, coh, cross_evil):
    return {
        "alpha": alpha, "trait_score": trait,
        "coherence_score": coh,
        "cross_trait_scores": {"evil": cross_evil},
    }


def test_compute_aggregates_basic():
    responses = [
        _resp(0.0, 30, 80, 10),
        _resp(0.0, 40, 90, 20),
        _resp(1.0, 70, 60, 30),
        _resp(1.0, 80, 50, 40),
    ]
    result = _compute_aggregates(responses, [0.0, 1.0])

    agg_0 = result["per_alpha"]["0.0"]
    assert agg_0["n_responses"] == 2
    assert abs(agg_0["trait_score_mean"] - 35.0) < 0.01
    assert abs(agg_0["coherence_score_mean"] - 85.0) < 0.01
    assert abs(agg_0["cross_trait_leakage"]["evil"]["mean"] - 15.0) < 0.01

    agg_1 = result["per_alpha"]["1.0"]
    assert agg_1["n_responses"] == 2
    assert abs(agg_1["trait_score_mean"] - 75.0) < 0.01


def test_compute_aggregates_empty():
    result = _compute_aggregates([], [0.0])
    assert result["per_alpha"] == {}


def test_compute_aggregates_none_scores():
    responses = [
        {"alpha": 0.0, "trait_score": None, "coherence_score": 80},
        {"alpha": 0.0, "trait_score": 50, "coherence_score": None},
    ]
    result = _compute_aggregates(responses, [0.0])
    agg = result["per_alpha"]["0.0"]
    assert agg["trait_score_mean"] == 50.0
    assert agg["coherence_score_mean"] == 80.0

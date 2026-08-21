"""Tests for frozen-response position re-extraction helpers."""

from deconfounding_interp.pipelines.position_reextraction import _group_response_rows


def test_group_response_rows_preserves_complete_polarities():
    rows = [
        {"side": "pos", "system_prompt": "p", "question": "q", "response": "a"},
        {"side": "neg", "system_prompt": "n", "question": "q", "response": "b"},
        {"side": "pos", "system_prompt": "bad", "question": "q"},
        {"side": "other", "system_prompt": "x", "question": "q", "response": "c"},
    ]
    grouped = _group_response_rows(rows)
    assert len(grouped["pos"]) == 1
    assert len(grouped["neg"]) == 1

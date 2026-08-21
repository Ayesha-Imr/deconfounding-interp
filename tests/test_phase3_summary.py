from deconfounding_interp.pipelines.phase3_summary import DIRECTION_TYPES


def test_phase3_summary_includes_causal_control_directions():
    assert "random" in DIRECTION_TYPES
    assert "sign_reversed" in DIRECTION_TYPES

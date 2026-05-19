from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.direction_reports import (
    build_direction_summary,
    random_surface_overlap_baseline,
    validate_trait_model_readiness,
)
from deconfounding_interp.directions import normalize
from deconfounding_interp.pipelines.base import StageContext
from deconfounding_interp.pipelines.direction_analysis import DirectionAnalysisStage


def _bundle(tmp_path: Path, trait_ids=("trait",), model_ids=("model",)):
    generation = {
        "system_prompt_pairs": 2,
        "paraphrases_per_prompt": 0,
    }
    return SimpleNamespace(
        project_root=tmp_path,
        experiment=SimpleNamespace(
            paths={
                "interim_dir": "data/interim",
                "direction_dir": "outputs/directions",
                "report_dir": "outputs/reports",
            },
            generation=generation,
            corrections={
                "direction_types": ["standard", "averaged", "subtracted", "single_variant"],
                "surface_basis": {"max_rank": 1, "variance_threshold": 0.90},
            },
            analysis={
                "stability": {
                    "within_trait_thresholds": {"high": 0.90, "moderate": 0.70},
                },
            },
            random_seed=17,
        ),
        models={
            model_id: SimpleNamespace(trait_layers={trait_id: 0 for trait_id in trait_ids})
            for model_id in model_ids
        },
        traits={
            trait_id: SimpleNamespace(expected_surface_confound="medium")
            for trait_id in trait_ids
        },
    )


def _write_activation_fixture(bundle, trait_id="trait", model_id="model", variant_count=2):
    for vi in range(variant_count):
        act_dir = (
            dio.trait_interim_dir(bundle, trait_id, model_id)
            / "activations"
            / f"variant_{vi:02d}"
        )
        pos = np.array([
            [2.0 + vi, 1.0, 0.0],
            [3.0 + vi, 1.0, 0.5],
        ])
        neg = np.array([
            [0.0, 1.5 + vi, 1.0],
            [0.0, 2.0 + vi, 1.0],
        ])
        dio.save_activations(act_dir, {0: {"pos": pos, "neg": neg}})


def test_direction_analysis_saves_single_variant_and_unit_normalized_outputs(tmp_path):
    bundle = _bundle(tmp_path)
    _write_activation_fixture(bundle, variant_count=2)
    stage = DirectionAnalysisStage()

    result = stage.run(
        {
            "model_id": "model",
            "trait_id": "trait",
            "payload": {
                "variant_count": 2,
                "direction_types": ["standard", "averaged", "subtracted", "single_variant"],
            },
        },
        StageContext(bundle=bundle, run_dir=tmp_path / "run"),
    )

    assert result["status"] == "completed"
    d_dir = dio.direction_dir(bundle, "trait", "model")
    expected = [
        "standard",
        "averaged",
        "subtracted",
        "single_variant",
        "variant_00",
        "variant_01",
    ]
    for name in expected:
        arr = np.load(d_dir / f"{name}.npy")
        np.testing.assert_allclose(np.linalg.norm(arr), 1.0)


def test_readiness_validation_reports_missing_variants_and_sides(tmp_path):
    bundle = _bundle(tmp_path)
    act_dir = dio.trait_interim_dir(bundle, "trait", "model") / "activations" / "variant_00"
    dio.save_activations(act_dir, {0: {"pos": np.ones((2, 3))}})

    report = validate_trait_model_readiness(bundle, "trait", "model", variant_count=2)

    assert report["status"] == "blocked"
    assert report["problem_count"] == 2
    assert report["missing_variant_indices"] == [0, 1]
    assert report["variants"][0]["missing_sides"] == ["neg"]


def test_random_surface_overlap_baseline_is_deterministic():
    surface = np.array([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]])

    first = random_surface_overlap_baseline(surface, hidden_dim=3, seed=123, n_samples=20)
    second = random_surface_overlap_baseline(surface, hidden_dim=3, seed=123, n_samples=20)

    assert first == second


def test_direction_summary_generation_writes_aggregate_reports(tmp_path):
    bundle = _bundle(tmp_path, trait_ids=("trait_a", "trait_b"))
    for trait_id, offset in [("trait_a", 0.0), ("trait_b", 1.0)]:
        d_dir = dio.direction_dir(bundle, trait_id, "model")
        dio.save_direction(d_dir, "standard", normalize(np.array([1.0, offset, 0.1])))
        dio.save_direction(d_dir, "single_variant", normalize(np.array([1.0, offset, 0.0])))
        dio.save_direction(d_dir, "averaged", normalize(np.array([1.0, offset, 0.2])))
        dio.save_direction(d_dir, "subtracted", normalize(np.array([0.0, 1.0, 0.3 + offset])))
        for vi in range(2):
            dio.save_direction(
                d_dir,
                f"variant_{vi:02d}",
                normalize(np.array([1.0, offset, 0.1 + vi])),
            )
        for si in range(2):
            dio.save_direction(
                d_dir,
                f"surface_{si:02d}",
                normalize(np.array([0.1 + si, 1.0, offset])),
            )
        _write_activation_fixture(bundle, trait_id=trait_id, variant_count=2)

    result = build_direction_summary(bundle, variant_count=2, write_figures=False)

    phase2_dir = tmp_path / "outputs" / "reports" / "phase2"
    assert result["status"] == "completed"
    assert (phase2_dir / "readiness.json").exists()
    assert (phase2_dir / "control_checks.json").exists()
    assert (phase2_dir / "stability_summary.csv").exists()
    assert (phase2_dir / "cross_trait_cosines.csv").exists()
    assert (phase2_dir / "surface_overlap_summary.csv").exists()
    assert (phase2_dir / "summary.md").exists()
    assert (phase2_dir / "direction_summary.json").exists()

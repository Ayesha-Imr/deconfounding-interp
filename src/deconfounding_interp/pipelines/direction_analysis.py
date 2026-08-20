"""Pipeline stage: compute DiM directions, surface-form directions, stability, and overlap."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.direction_reports import (
    resolve_selected_layer,
    validate_trait_model_readiness,
)
from deconfounding_interp.analysis.stability import summarize_stability
from deconfounding_interp.analysis.surface_overlap import compute_surface_overlap
from deconfounding_interp.directions import (
    average_directions,
    difference_in_means,
    normalize,
    orthonormal_basis,
)
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)


class DirectionAnalysisStage:
    name = "direction_analysis"

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info(
                "[DRY RUN] Would run direction analysis for "
                "model=%s trait=%s", job["model_id"], job["trait_id"],
            )
            return {"status": "dry_run"}

        bundle = context.bundle
        model_id = job["model_id"]
        trait_id = job["trait_id"]
        payload = job["payload"]
        variant_count = payload["variant_count"]

        corrections_cfg = bundle.experiment.corrections
        direction_types = set(
            payload.get(
                "direction_types",
                corrections_cfg.get("direction_types", ["standard", "averaged", "subtracted"]),
            )
        )

        # Determine best layer
        r_dir = dio.report_dir(bundle, trait_id, model_id)
        payload_layer = payload.get("selected_layer")
        try:
            if payload_layer is not None:
                selected_layer = int(payload_layer)
            else:
                selected_layer, layer_source = resolve_selected_layer(
                    bundle,
                    model_id,
                    trait_id,
                    variant_count,
                )
                if selected_layer is None:
                    raise RuntimeError(
                        f"No usable layer found for trait={trait_id} model={model_id}; "
                        f"last checked {layer_source}"
                    )
                selected_layer = int(selected_layer)
        except RuntimeError as exc:
            readiness = validate_trait_model_readiness(
                bundle,
                trait_id=trait_id,
                model_id=model_id,
                variant_count=variant_count,
            )
            dio.save_results_json(r_dir / "readiness.json", readiness)
            logger.warning(
                "Skipping direction analysis for trait=%s model=%s: %s",
                trait_id,
                model_id,
                exc,
            )
            return _blocked_result(readiness, r_dir / "readiness.json", str(exc))
        logger.info("Using layer %d for trait=%s model=%s", selected_layer, trait_id, model_id)

        readiness = validate_trait_model_readiness(
            bundle,
            trait_id=trait_id,
            model_id=model_id,
            variant_count=variant_count,
            selected_layer=selected_layer,
        )
        dio.save_results_json(r_dir / "readiness.json", readiness)
        if readiness["status"] != "ready" and payload_layer is not None:
            fallback_layer, _ = resolve_selected_layer(
                bundle,
                model_id,
                trait_id,
                variant_count,
            )
            if fallback_layer is not None and int(fallback_layer) != selected_layer:
                selected_layer = int(fallback_layer)
                logger.info(
                    "Payload layer was incomplete; using fallback layer %d for trait=%s model=%s",
                    selected_layer,
                    trait_id,
                    model_id,
                )
                readiness = validate_trait_model_readiness(
                    bundle,
                    trait_id=trait_id,
                    model_id=model_id,
                    variant_count=variant_count,
                    selected_layer=selected_layer,
                )
                dio.save_results_json(r_dir / "readiness.json", readiness)
        if readiness["status"] != "ready":
            reason = (
                "Phase 2 inputs are not ready for "
                f"trait={trait_id} model={model_id}; see {r_dir / 'readiness.json'}"
            )
            logger.warning(reason)
            return _blocked_result(readiness, r_dir / "readiness.json", reason)

        # Load activations for all variants at the selected layer
        interim = dio.trait_interim_dir(bundle, trait_id, model_id)
        variant_acts = {}
        for vi in range(variant_count):
            act_dir = interim / "activations" / f"variant_{vi:02d}"
            acts = dio.load_activations(act_dir, layer=selected_layer)
            if selected_layer in acts:
                variant_acts[vi] = acts[selected_layer]

        if not variant_acts:
            raise RuntimeError(
                f"No activations found for trait={trait_id} "
                f"model={model_id} layer={selected_layer}"
            )

        # Compute per-variant DiM directions
        variant_raw_directions = {}
        variant_directions = {}
        for vi, sides in variant_acts.items():
            if "pos" in sides and "neg" in sides:
                raw_direction = difference_in_means(sides["pos"], sides["neg"])
                variant_raw_directions[vi] = raw_direction
                variant_directions[vi] = normalize(raw_direction)

        if not variant_directions:
            raise RuntimeError("No valid variant directions could be computed")

        # v_standard: pool all original variants (indices 0-4)
        n_originals = min(5, variant_count)
        pos_all = np.concatenate([
            variant_acts[vi]["pos"] for vi in range(n_originals)
            if vi in variant_acts and "pos" in variant_acts[vi]
        ])
        neg_all = np.concatenate([
            variant_acts[vi]["neg"] for vi in range(n_originals)
            if vi in variant_acts and "neg" in variant_acts[vi]
        ])
        v_standard_raw = difference_in_means(pos_all, neg_all)
        v_standard = normalize(v_standard_raw)

        # Surface-form directions: pair same-side variants
        surface_dirs = _compute_surface_directions(variant_acts)

        # Stability analysis
        dir_array = np.array([variant_directions[vi] for vi in sorted(variant_directions)])
        stability = summarize_stability(dir_array)
        logger.info(
            "Stability: mean_cosine=%.3f std=%.3f",
            stability.mean_cosine, stability.std_cosine,
        )

        # Surface overlap
        surface_basis_cfg = corrections_cfg.get("surface_basis", {})
        max_rank = surface_basis_cfg.get("max_rank", 5)
        variance_threshold = surface_basis_cfg.get("variance_threshold", 0.90)

        overlap = None
        if len(surface_dirs) >= 2:
            overlap = compute_surface_overlap(
                v_standard, surface_dirs,
                max_rank=max_rank,
                variance_threshold=variance_threshold,
            )
            logger.info("Surface overlap: %.3f", overlap.overlap_fraction)

        # Corrected directions
        v_averaged = average_directions(dir_array)
        v_averaged_raw = np.mean(
            np.array([variant_raw_directions[vi] for vi in sorted(variant_raw_directions)]),
            axis=0,
        )
        v_single_variant = variant_directions[0]
        v_single_variant_raw = variant_raw_directions[0]
        v_subtracted_raw = v_standard_raw.copy()
        v_subtracted = v_standard
        if len(surface_dirs) >= 2:
            basis = orthonormal_basis(
                np.array(surface_dirs),
                max_rank=max_rank,
                variance_threshold=variance_threshold,
            )
            v_subtracted_raw = v_standard_raw - (basis.T @ (basis @ v_standard_raw))
            v_subtracted = normalize(v_subtracted_raw)

        # Save directions
        d_dir = dio.direction_dir(bundle, trait_id, model_id)
        configured_directions = {
            "standard": v_standard,
            "averaged": v_averaged,
            "subtracted": v_subtracted,
            "single_variant": v_single_variant,
        }
        raw_directions = {
            "standard": v_standard_raw,
            "averaged": v_averaged_raw,
            "subtracted": v_subtracted_raw,
            "single_variant": v_single_variant_raw,
        }

        # Save a leave-one-variant-out fit for the held-out probing variant.
        # The old pipeline used the all-variant averaged/subtracted directions
        # to score the last variant, which leaks the probe labels into the fit.
        holdout_idx = int(payload.get("probe_holdout_index", variant_count - 1))
        fit_directions, fit_raw_directions = _fit_direction_set(
            variant_acts,
            excluded_variant=holdout_idx,
            n_originals=n_originals,
            max_rank=max_rank,
            variance_threshold=variance_threshold,
        )
        for name, direction in configured_directions.items():
            if name in direction_types:
                dio.save_direction(d_dir, name, direction)
                dio.save_direction(d_dir, f"{name}_raw", raw_directions[name])
                if name in fit_directions:
                    fit_name = f"{name}_fit_excluding_variant_{holdout_idx:02d}"
                    dio.save_direction(d_dir, fit_name, fit_directions[name])
                    dio.save_direction(
                        d_dir,
                        f"{fit_name}_raw",
                        fit_raw_directions[name],
                    )
        for vi, d in variant_directions.items():
            dio.save_direction(d_dir, f"variant_{vi:02d}", d)
            dio.save_direction(d_dir, f"variant_{vi:02d}_raw", variant_raw_directions[vi])
        for si, sd in enumerate(surface_dirs):
            dio.save_direction(d_dir, f"surface_{si:02d}", sd)

        # Save selected layer info
        dio.save_results_json(d_dir / "selected_layer.json", {"best_layer": selected_layer})

        # Save analysis reports
        dio.save_results_json(r_dir / "stability.json", asdict(stability))
        if overlap is not None:
            dio.save_results_json(r_dir / "surface_overlap.json", asdict(overlap))
        else:
            dio.save_results_json(
                r_dir / "surface_overlap.json",
                {
                    "status": "unavailable",
                    "reason": "Need at least two surface directions",
                    "n_surface_dirs": len(surface_dirs),
                },
            )
        dio.save_results_json(
            d_dir / "direction_metadata.json",
            {
                "configured_direction_types": sorted(direction_types),
                "saved_direction_types": sorted(
                    name for name in configured_directions if name in direction_types
                ),
                "saved_variant_directions": sorted(
                    f"variant_{vi:02d}" for vi in variant_directions
                ),
                "saved_surface_directions": [
                    f"surface_{si:02d}" for si in range(len(surface_dirs))
                ],
                "unit_normalized": True,
                "raw_direction_files": {
                    name: f"{name}_raw.npy"
                    for name in configured_directions
                    if name in direction_types
                },
                "raw_direction_norms": {
                    name: float(np.linalg.norm(raw_directions[name]))
                    for name in configured_directions
                    if name in direction_types
                },
                "probe_holdout_index": holdout_idx,
                "leakage_safe_direction_files": {
                    name: f"{name}_fit_excluding_variant_{holdout_idx:02d}.npy"
                    for name in fit_directions
                    if name in direction_types
                },
            },
        )

        return {
            "status": "completed",
            "selected_layer": selected_layer,
            "n_variants": len(variant_directions),
            "n_surface_dirs": len(surface_dirs),
            "mean_cosine": stability.mean_cosine,
        }


def _compute_surface_directions(
    variant_acts: dict[int, dict[str, np.ndarray]],
) -> list[np.ndarray]:
    """Compute DiM between same-side prompt variants (pure wording artifact)."""
    surface_dirs = []
    variant_indices = sorted(variant_acts.keys())

    for side in ("pos", "neg"):
        indices_with_side = [vi for vi in variant_indices if side in variant_acts[vi]]
        for vi_a, vi_b in itertools.combinations(indices_with_side, 2):
            acts_a = variant_acts[vi_a][side]
            acts_b = variant_acts[vi_b][side]
            surface_dirs.append(normalize(difference_in_means(acts_a, acts_b)))

    return surface_dirs


def _fit_direction_set(
    variant_acts: dict[int, dict[str, np.ndarray]],
    *,
    excluded_variant: int,
    n_originals: int,
    max_rank: int,
    variance_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit all direction variants without using one explicitly held-out variant."""
    fit_acts = {
        vi: sides for vi, sides in variant_acts.items() if vi != excluded_variant
    }
    raw_by_variant: dict[int, np.ndarray] = {}
    for vi, sides in fit_acts.items():
        if "pos" in sides and "neg" in sides:
            raw = difference_in_means(sides["pos"], sides["neg"])
            raw_by_variant[vi] = raw
    if not raw_by_variant:
        return {}, {}

    original_indices = [
        vi for vi in range(n_originals)
        if vi in fit_acts and "pos" in fit_acts[vi] and "neg" in fit_acts[vi]
    ]
    if not original_indices:
        return {}, {}
    pos_all = np.concatenate([fit_acts[vi]["pos"] for vi in original_indices])
    neg_all = np.concatenate([fit_acts[vi]["neg"] for vi in original_indices])
    standard_raw = difference_in_means(pos_all, neg_all)
    surface_dirs = _compute_surface_directions(fit_acts)

    averaged_raw = np.mean(np.array(list(raw_by_variant.values())), axis=0)
    subtracted_raw = standard_raw.copy()
    if len(surface_dirs) >= 2:
        basis = orthonormal_basis(
            np.array(surface_dirs),
            max_rank=max_rank,
            variance_threshold=variance_threshold,
        )
        subtracted_raw = standard_raw - (basis.T @ (basis @ standard_raw))

    single_index = 0 if 0 in raw_by_variant else min(raw_by_variant)
    raw = {
        "standard": standard_raw,
        "averaged": averaged_raw,
        "subtracted": subtracted_raw,
        "single_variant": raw_by_variant[single_index],
    }
    unit = {name: normalize(value) for name, value in raw.items()}
    return unit, raw


def _blocked_result(readiness: dict[str, Any], report_path, reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "readiness_report": str(report_path),
        "selected_layer": readiness.get("selected_layer"),
        "variant_count_ready": readiness.get("variant_count_ready"),
        "variant_count_expected": readiness.get("variant_count_expected"),
        "problem_count": readiness.get("problem_count"),
    }

"""Pipeline stage: compute DiM directions, surface-form directions, stability, and overlap."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.stability import summarize_stability
from deconfounding_interp.analysis.surface_overlap import compute_surface_overlap
from deconfounding_interp.directions import (
    average_directions,
    difference_in_means,
    orthonormal_basis,
    remove_subspace,
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

        # Determine best layer
        selected_layer = _resolve_layer(bundle, model_id, trait_id)
        logger.info("Using layer %d for trait=%s model=%s", selected_layer, trait_id, model_id)

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
        variant_directions = {}
        for vi, sides in variant_acts.items():
            if "pos" in sides and "neg" in sides:
                variant_directions[vi] = difference_in_means(sides["pos"], sides["neg"])

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
        v_standard = difference_in_means(pos_all, neg_all)

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
        v_subtracted = v_standard
        if len(surface_dirs) >= 2:
            basis = orthonormal_basis(
                np.array(surface_dirs),
                max_rank=max_rank,
                variance_threshold=variance_threshold,
            )
            v_subtracted = remove_subspace(v_standard, basis)

        # Save directions
        d_dir = dio.direction_dir(bundle, trait_id, model_id)
        dio.save_direction(d_dir, "standard", v_standard)
        dio.save_direction(d_dir, "averaged", v_averaged)
        dio.save_direction(d_dir, "subtracted", v_subtracted)
        for vi, d in variant_directions.items():
            dio.save_direction(d_dir, f"variant_{vi:02d}", d)
        for si, sd in enumerate(surface_dirs):
            dio.save_direction(d_dir, f"surface_{si:02d}", sd)

        # Save selected layer info
        dio.save_results_json(d_dir / "selected_layer.json", {"best_layer": selected_layer})

        # Save analysis reports
        r_dir = dio.report_dir(bundle, trait_id, model_id)
        dio.save_results_json(r_dir / "stability.json", asdict(stability))
        if overlap is not None:
            dio.save_results_json(r_dir / "surface_overlap.json", asdict(overlap))

        return {
            "status": "completed",
            "selected_layer": selected_layer,
            "n_variants": len(variant_directions),
            "n_surface_dirs": len(surface_dirs),
            "mean_cosine": stability.mean_cosine,
        }


def _resolve_layer(bundle, model_id: str, trait_id: str) -> int:
    model_config = bundle.models[model_id]
    configured = model_config.trait_layers.get(trait_id)
    if configured is not None:
        return configured

    # Try loading from AUROC sweep result
    interim = dio.trait_interim_dir(bundle, trait_id, model_id)
    sweep_path = interim / "selected_layer.json"
    if sweep_path.exists():
        data = dio.load_results_json(sweep_path)
        return data["best_layer"]

    raise RuntimeError(
        f"No layer configured for trait={trait_id} model={model_id} "
        f"and no AUROC sweep result found at {sweep_path}"
    )


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
            surface_dirs.append(difference_in_means(acts_a, acts_b))

    return surface_dirs

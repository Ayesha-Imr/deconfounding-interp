"""Pipeline stage for split-half and label-shuffle null calibration."""

from __future__ import annotations

import logging
from typing import Any

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.direction_reports import resolve_selected_layer
from deconfounding_interp.analysis.nulls import run_null_audit
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)


class NullAnalysisStage:
    name = "null_analysis"

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info(
                "[DRY RUN] Would run null analysis for model=%s trait=%s",
                job["model_id"], job["trait_id"],
            )
            return {"status": "dry_run"}

        bundle = context.bundle
        model_id = job["model_id"]
        trait_id = job["trait_id"]
        payload = job.get("payload", {})
        variant_count = int(payload.get("variant_count", 10))
        selected_layer = payload.get("selected_layer")
        if selected_layer is None:
            selected_layer, _ = resolve_selected_layer(
                bundle, model_id, trait_id, variant_count,
            )
        if selected_layer is None:
            return {"status": "blocked", "reason": "no_selected_layer"}
        selected_layer = int(selected_layer)

        interim = dio.trait_interim_dir(bundle, trait_id, model_id)
        variant_activations = {}
        for vi in range(variant_count):
            acts = dio.load_activations(
                interim / "activations" / f"variant_{vi:02d}",
                layer=selected_layer,
            )
            sides = acts.get(selected_layer)
            if sides and "pos" in sides and "neg" in sides:
                variant_activations[vi] = sides
        if len(variant_activations) < 2:
            return {
                "status": "blocked",
                "reason": "fewer_than_two_complete_variants",
                "n_complete_variants": len(variant_activations),
            }

        repeats = int(
            payload.get(
                "repeats",
                bundle.experiment.analysis.get("nulls", {}).get("repeats", 90),
            )
        )
        result = run_null_audit(
            variant_activations,
            repeats=repeats,
            seed=bundle.experiment.random_seed,
        )
        result.update({
            "model_id": model_id,
            "trait_id": trait_id,
            "selected_layer": selected_layer,
            "complete_variant_indices": sorted(variant_activations),
            "n_complete_variants": len(variant_activations),
        })
        out_path = (
            dio.resolve_paths(bundle)["report_dir"]
            / "phase2" / "nulls" / f"{model_id}__{trait_id}.json"
        )
        dio.save_results_json(out_path, result)
        logger.info(
            "Null analysis %s/%s: split-half=%s label-shuffle=%s",
            model_id, trait_id,
            result["split_half"]["summary"]["status"],
            result["label_shuffle"]["summary"]["status"],
        )
        return {
            "status": "completed",
            "selected_layer": selected_layer,
            "n_complete_variants": len(variant_activations),
            "output": str(out_path),
        }

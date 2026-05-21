"""Pipeline stage: probe held-out activations with each direction type (CPU-only)."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.probing import probe_with_direction
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)

DIRECTION_TYPES = ("standard", "averaged", "subtracted", "single_variant")


class ProbingStage:
    name = "probing"

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info(
                "[DRY RUN] Would run probing for model=%s trait=%s",
                job["model_id"], job["trait_id"],
            )
            return {"status": "dry_run"}

        t0 = time.time()
        bundle = context.bundle
        model_id = job["model_id"]
        trait_id = job["trait_id"]
        payload = job["payload"]

        d_dir = dio.direction_dir(bundle, trait_id, model_id)

        # Resolve selected layer
        selected_layer = payload.get("selected_layer")
        if selected_layer is None:
            layer_path = d_dir / "selected_layer.json"
            if layer_path.exists():
                selected_layer = dio.load_results_json(layer_path)["best_layer"]
        if selected_layer is None:
            logger.warning("No selected layer for %s/%s, skipping probing", trait_id, model_id)
            return {"status": "blocked", "reason": "no_selected_layer"}
        selected_layer = int(selected_layer)

        # Load held-out activations (last variant)
        variant_count = payload.get("variant_count", 10)
        holdout_idx = variant_count - 1
        act_dir = (
            dio.trait_interim_dir(bundle, trait_id, model_id)
            / "activations" / f"variant_{holdout_idx:02d}"
        )
        acts = dio.load_activations(act_dir, layer=selected_layer)
        sides = acts.get(selected_layer, {})

        if "pos" not in sides or "neg" not in sides:
            logger.warning(
                "Missing held-out activations for %s/%s variant_%02d layer=%d",
                trait_id, model_id, holdout_idx, selected_layer,
            )
            return {"status": "blocked", "reason": "missing_holdout_activations"}

        pos_acts = sides["pos"]
        neg_acts = sides["neg"]

        logger.info(
            "Probing %s/%s: layer=%d, holdout=variant_%02d, "
            "pos=%d neg=%d samples",
            trait_id, model_id, selected_layer, holdout_idx,
            pos_acts.shape[0], neg_acts.shape[0],
        )

        direction_types = payload.get("direction_types", list(DIRECTION_TYPES))
        results = {}
        for dt in direction_types:
            npy_path = d_dir / f"{dt}.npy"
            if not npy_path.exists():
                logger.info("  %s: direction not found, skipping", dt)
                continue

            direction = np.load(npy_path)
            probe = probe_with_direction(pos_acts, neg_acts, direction)
            results[dt] = asdict(probe)
            logger.info(
                "  %s: AUROC=%.4f accuracy=%.4f",
                dt, probe.auroc, probe.accuracy,
            )

        out_dir = dio.resolve_paths(bundle)["report_dir"] / "phase3" / trait_id / model_id
        dio.save_results_json(out_dir / "probing_results.json", results)

        elapsed = time.time() - t0
        logger.info(
            "Probing complete for %s/%s (%d direction types, %.1fs)",
            trait_id, model_id, len(results), elapsed,
        )
        return {
            "status": "completed",
            "direction_types_probed": list(results.keys()),
            "elapsed_seconds": round(elapsed, 1),
        }

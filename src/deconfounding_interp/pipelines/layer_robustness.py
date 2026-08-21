"""Pipeline stage for leakage-safe all-layer probing curves."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.probing import probe_with_direction
from deconfounding_interp.directions import difference_in_means, normalize
from deconfounding_interp.pipelines.base import StageContext
from deconfounding_interp.pipelines.direction_controls import stable_control_seed

logger = logging.getLogger(__name__)


class LayerRobustnessStage:
    """Fit on training prompt variants and probe every residual-stream layer."""

    name = "layer_robustness"

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info(
                "[DRY RUN] Would run all-layer probing for model=%s trait=%s",
                job["model_id"], job["trait_id"],
            )
            return {"status": "dry_run"}

        t0 = time.time()
        bundle = context.bundle
        model_id = job["model_id"]
        trait_id = job["trait_id"]
        payload = job.get("payload", {})
        variant_count = int(payload.get("variant_count", 5))
        holdout_idx = int(payload.get("holdout_index", variant_count - 1))
        train_indices = [
            int(index) for index in payload.get(
                "train_variant_indices",
                [index for index in range(variant_count) if index != holdout_idx],
            )
        ]

        interim = dio.trait_interim_dir(bundle, trait_id, model_id)
        train_by_layer: dict[int, dict[str, list[np.ndarray]]] = {}
        for variant_idx in train_indices:
            acts = dio.load_activations(
                interim / "activations" / f"variant_{variant_idx:02d}",
                layer=None,
            )
            for layer, sides in acts.items():
                if "pos" in sides and "neg" in sides:
                    train_by_layer.setdefault(layer, {"pos": [], "neg": []})
                    train_by_layer[layer]["pos"].append(sides["pos"])
                    train_by_layer[layer]["neg"].append(sides["neg"])

        holdout = dio.load_activations(
            interim / "activations" / f"variant_{holdout_idx:02d}",
            layer=None,
        )
        result = compute_layer_results(
            train_by_layer=train_by_layer,
            holdout=holdout,
            base_seed=bundle.experiment.random_seed,
            model_id=model_id,
            trait_id=trait_id,
            train_variant_indices=train_indices,
            holdout_index=holdout_idx,
        )
        out_path = (
            dio.resolve_paths(bundle)["report_dir"]
            / "phase4" / trait_id / model_id / "layer_probing.json"
        )
        dio.save_results_json(out_path, result)
        elapsed = time.time() - t0
        if result["status"] != "completed":
            return {"status": "blocked", "reason": result["reason"], "output": str(out_path)}
        logger.info(
            "Layer robustness %s/%s: %d layers (%.1fs)",
            trait_id, model_id, len(result["layers"]), elapsed,
        )
        return {
            "status": "completed",
            "n_layers": len(result["layers"]),
            "output": str(out_path),
            "elapsed_seconds": round(elapsed, 1),
        }


def compute_layer_results(
    *,
    train_by_layer: dict[int, dict[str, list[np.ndarray]]],
    holdout: dict[int, dict[str, np.ndarray]],
    base_seed: int,
    model_id: str,
    trait_id: str,
    train_variant_indices: list[int],
    holdout_index: int,
) -> dict[str, Any]:
    """Return all-layer standard/control probe results without filesystem I/O."""

    rows: list[dict[str, Any]] = []
    for layer in sorted(set(train_by_layer) & set(holdout)):
        train_sides = train_by_layer[layer]
        test_sides = holdout[layer]
        if not train_sides["pos"] or not train_sides["neg"]:
            continue
        if "pos" not in test_sides or "neg" not in test_sides:
            continue
        train_pos = np.concatenate(train_sides["pos"])
        train_neg = np.concatenate(train_sides["neg"])
        standard = normalize(difference_in_means(train_pos, train_neg))
        random_seed = stable_control_seed(base_seed, model_id, trait_id)
        random_direction = normalize(
            np.random.default_rng(random_seed).normal(size=standard.shape)
        )
        directions = {
            "standard": standard,
            "random": random_direction,
            "sign_reversed": -standard,
        }
        row: dict[str, Any] = {
            "layer": int(layer),
            "train_variant_indices": list(train_variant_indices),
            "holdout_index": int(holdout_index),
            "n_train_pos": int(train_pos.shape[0]),
            "n_train_neg": int(train_neg.shape[0]),
            "n_holdout_pos": int(test_sides["pos"].shape[0]),
            "n_holdout_neg": int(test_sides["neg"].shape[0]),
            "random_seed": random_seed,
            "directions": {},
        }
        for direction_type, direction in directions.items():
            probe = probe_with_direction(
                test_sides["pos"], test_sides["neg"], direction,
            )
            row["directions"][direction_type] = {
                "auroc": float(probe.auroc),
                "accuracy": float(probe.accuracy),
                "direction_norm": float(np.linalg.norm(direction)),
            }
        rows.append(row)

    if not rows:
        return {
            "status": "blocked",
            "reason": "no_complete_train_holdout_layers",
            "train_variant_indices": list(train_variant_indices),
            "holdout_index": int(holdout_index),
            "layers": [],
        }
    return {
        "status": "completed",
        "train_variant_indices": list(train_variant_indices),
        "holdout_index": int(holdout_index),
        "layers": rows,
    }

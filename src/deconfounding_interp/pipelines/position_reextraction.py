"""Re-extract existing responses at alternate activation positions."""

from __future__ import annotations

import logging
import time
from typing import Any

from deconfounding_interp import io as dio
from deconfounding_interp.backends import create_backend
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)


class PositionReextractionStage:
    """Reuse frozen response text and extract prompt-position activations."""

    name = "position_reextraction"

    def __init__(self):
        self._backend = None
        self._loaded_model_id: str | None = None

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info(
                "[DRY RUN] Would re-extract positions for model=%s trait=%s",
                job["model_id"], job["trait_id"],
            )
            return {"status": "dry_run"}

        t0 = time.time()
        bundle = context.bundle
        model_id = job["model_id"]
        trait_id = job["trait_id"]
        payload = job.get("payload", {})
        positions = [str(position) for position in payload.get("positions", [])]
        variant_count = int(payload.get("variant_count", 5))
        source_root = bundle.project_root / str(payload["source_interim_root"])
        output_root = bundle.project_root / str(payload["output_interim_root"])
        source_dir = source_root / trait_id / model_id / "responses"
        if not positions:
            return {"status": "blocked", "reason": "no_positions_configured"}

        model_config = bundle.models[model_id]
        backend_name = model_config.backend or bundle.experiment.backend
        if self._backend is None or self._loaded_model_id != model_id:
            if self._backend is not None:
                self._backend.unload_model()
            self._backend = create_backend(backend_name)
            self._backend.load_model(model_config)
            self._loaded_model_id = model_id

        counts: dict[str, dict[str, int]] = {}
        for variant_idx in range(variant_count):
            response_path = source_dir / f"variant_{variant_idx:02d}.json"
            if not response_path.exists():
                return {
                    "status": "blocked",
                    "reason": "missing_source_responses",
                    "variant_index": variant_idx,
                    "path": str(response_path),
                }
            rows = dio.load_responses_json(response_path)
            by_side = _group_response_rows(rows)
            if not by_side["pos"] or not by_side["neg"]:
                return {
                    "status": "blocked",
                    "reason": "missing_source_polarity",
                    "variant_index": variant_idx,
                }

            for position in positions:
                position_counts = counts.setdefault(position, {"pos": 0, "neg": 0})
                for side in ("pos", "neg"):
                    side_rows = by_side[side]
                    texts = [
                        self._backend.format_chat(
                            row["system_prompt"], row["question"], row["response"],
                        )
                        for row in side_rows
                    ]
                    prompt_lengths = [
                        self._backend.get_prompt_token_length(
                            row["system_prompt"], row["question"],
                        )
                        for row in side_rows
                    ]
                    activations = self._backend.extract_activations(
                        texts, prompt_lengths, position=position,
                    )
                    output_dir = (
                        output_root / trait_id / model_id / "activations"
                        / position / f"variant_{variant_idx:02d}"
                    )
                    dio.save_activations(
                        output_dir,
                        {layer: {side: values} for layer, values in activations.items()},
                    )
                    position_counts[side] += len(side_rows)

        out_path = (
            dio.resolve_paths(bundle)["report_dir"]
            / "phase4" / trait_id / model_id / "position_reextraction.json"
        )
        dio.save_results_json(
            out_path,
            {
                "status": "completed",
                "positions": positions,
                "variant_count": variant_count,
                "source_interim_root": str(payload["source_interim_root"]),
                "output_interim_root": str(payload["output_interim_root"]),
                "response_counts": counts,
            },
        )
        elapsed = time.time() - t0
        logger.info(
            "Position re-extraction %s/%s: %s (%.1fs)",
            trait_id, model_id, positions, elapsed,
        )
        return {
            "status": "completed",
            "positions": positions,
            "output": str(out_path),
            "elapsed_seconds": round(elapsed, 1),
        }


def _group_response_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return complete polarity buckets while preserving response order."""

    result = {"pos": [], "neg": []}
    for row in rows:
        side = row.get("side")
        if side in result and all(
            key in row for key in ("system_prompt", "question", "response")
        ):
            result[side].append(row)
    return result

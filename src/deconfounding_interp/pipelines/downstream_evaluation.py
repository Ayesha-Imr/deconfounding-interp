"""Pipeline stage: steering comparison — generate steered responses and score them."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.backends import create_backend
from deconfounding_interp.directions import calibrate_steering_scale
from deconfounding_interp.llm.base import create_client
from deconfounding_interp.llm.coherence_judge import CoherenceJudge
from deconfounding_interp.llm.judge import TraitJudge
from deconfounding_interp.pipelines.base import StageContext
from deconfounding_interp.pipelines.direction_controls import resolve_direction

logger = logging.getLogger(__name__)


class DownstreamEvaluationStage:
    name = "downstream_evaluation"

    def __init__(self):
        self._backend = None
        self._loaded_model_id: str | None = None

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info(
                "[DRY RUN] Would run downstream eval for model=%s trait=%s direction=%s",
                job["model_id"], job["trait_id"],
                job["payload"].get("direction_type"),
            )
            return {"status": "dry_run"}
        return asyncio.run(self._run_async(job, context))

    async def _run_async(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        t0 = time.time()
        bundle = context.bundle
        model_id = job["model_id"]
        trait_id = job["trait_id"]
        payload = job["payload"]
        direction_type = payload["direction_type"]
        alpha_values = payload.get("alpha_values", [])

        if not alpha_values:
            alpha_values = bundle.experiment.steering.get(
                "alpha_values", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            )

        # Load a fitted direction or synthesize a deterministic causal control.
        d_dir = dio.direction_dir(bundle, trait_id, model_id)
        direction, direction_metadata = resolve_direction(
            d_dir,
            direction_type,
            base_seed=bundle.experiment.random_seed,
            model_id=model_id,
            trait_id=trait_id,
        )
        if direction is None:
            logger.warning(
                "Direction %s not found at %s, skipping",
                direction_type,
                direction_metadata["direction_source"],
            )
            return {"status": "blocked", "reason": "direction_not_found"}

        # Resolve layer
        selected_layer = payload.get("selected_layer")
        if selected_layer is None:
            layer_path = d_dir / "selected_layer.json"
            if layer_path.exists():
                selected_layer = dio.load_results_json(layer_path)["best_layer"]
        if selected_layer is None:
            logger.warning("No selected layer for %s/%s", trait_id, model_id)
            return {"status": "blocked", "reason": "no_selected_layer"}
        selected_layer = int(selected_layer)

        steering_cfg = bundle.experiment.steering
        scale_mode = steering_cfg.get("scale_mode", "unit")
        direction_scale = 1.0
        calibration_indices = steering_cfg.get("calibration_variant_indices", [0])
        if scale_mode == "activation_rms":
            calibration_acts = []
            interim = dio.trait_interim_dir(bundle, trait_id, model_id)
            for variant_idx in calibration_indices:
                act_dir = interim / "activations" / f"variant_{int(variant_idx):02d}"
                acts = dio.load_activations(act_dir, layer=selected_layer)
                sides = acts.get(selected_layer, {})
                calibration_acts.extend(
                    sides[side] for side in ("pos", "neg") if side in sides
                )
            if not calibration_acts:
                logger.warning(
                    "No calibration activations for %s/%s layer=%d",
                    trait_id, model_id, selected_layer,
                )
                return {"status": "blocked", "reason": "missing_scale_calibration"}
            direction_scale = calibrate_steering_scale(
                direction,
                np.concatenate(calibration_acts),
                target_rms_ratio=float(
                    steering_cfg.get("target_activation_rms_ratio", 0.05)
                ),
            )
        elif scale_mode != "unit":
            raise ValueError(f"Unknown steering scale_mode: {scale_mode!r}")

        # Load model
        model_config = bundle.models[model_id]
        backend_name = model_config.backend or bundle.experiment.backend
        if self._backend is None or self._loaded_model_id != model_id:
            if self._backend is not None:
                self._backend.unload_model()
            self._backend = create_backend(backend_name)
            self._backend.load_model(model_config)
            self._loaded_model_id = model_id

        # Load evaluation questions
        assets = dio.load_results_json(
            dio.trait_raw_dir(bundle, trait_id) / "assets.json"
        )
        eval_questions = assets["evaluation_questions"]

        # Set up judges only for the scientific scoring path. A judge-free
        # steering smoke still records every response and the exact scale,
        # which validates the hook without silently inventing behavior scores.
        score_mode = bundle.experiment.scoring.get("mode", "judge")
        if score_mode == "judge":
            llm_cfg = bundle.experiment.llm
            judge_client = create_client(
                provider=llm_cfg.get("provider", "openai"),
                model=llm_cfg.get("judge_model", "gpt-4.1-mini-2025-04-14"),
            )
            trait_judge = TraitJudge(judge_client, bundle.judge)
            coherence_judge = CoherenceJudge(judge_client)
            concurrency = llm_cfg.get("judge_concurrency", 50)
        elif score_mode == "none":
            trait_judge = None
            coherence_judge = None
            concurrency = 1
            logger.info(
                "Skipping steering judges for pilot: retaining generated responses",
            )
        else:
            raise ValueError(f"Unknown scoring.mode: {score_mode!r}")

        rollouts_per_q = steering_cfg.get("rollouts_per_eval_question", 10)
        max_new_tokens = model_config.default_max_new_tokens

        logger.info(
            "Starting steering eval: %s/%s direction=%s layer=%d "
            "alphas=%s rollouts_per_q=%d eval_questions=%d",
            trait_id, model_id, direction_type, selected_layer,
            alpha_values, rollouts_per_q, len(eval_questions),
        )

        all_responses: list[dict[str, Any]] = []

        for alpha_idx, alpha in enumerate(alpha_values):
            t_alpha = time.time()
            condition_responses = []

            for q in eval_questions:
                prompts = [{"system_prompt": "", "question": q}] * rollouts_per_q
                responses = self._backend.generate_with_steering(
                    prompts,
                    direction=direction,
                    layer=selected_layer,
                    alpha=alpha,
                    direction_scale=direction_scale,
                    max_new_tokens=max_new_tokens,
                )
                for resp in responses:
                    condition_responses.append({
                        "question": q,
                        "response": resp,
                        "alpha": alpha,
                        "direction_type": direction_type,
                        "direction_scale": direction_scale,
                        "scale_mode": scale_mode,
                        **direction_metadata,
                    })

            gen_time = time.time() - t_alpha
            n_responses = len(condition_responses)
            logger.info(
                "  alpha=%.1f: generated %d responses (%.1fs, %.2fs/resp)",
                alpha, n_responses, gen_time,
                gen_time / max(n_responses, 1),
            )

            if score_mode == "judge":
                # Score: trait expression
                t_score = time.time()
                trait = bundle.traits[trait_id]
                trait_results = await trait_judge.score_batch(
                    condition_responses, trait, concurrency=concurrency,
                )
                for r, jr in zip(condition_responses, trait_results, strict=True):
                    r["trait_score"] = jr.score

                # Score: coherence
                coherence_results = await coherence_judge.score_batch(
                    condition_responses, concurrency=concurrency,
                )
                for r, cr in zip(condition_responses, coherence_results, strict=True):
                    r["coherence_score"] = cr.score

                # Score: cross-trait leakage
                if steering_cfg.get("evaluate_cross_trait_leakage", True):
                    for other_trait_id, other_trait in bundle.traits.items():
                        if other_trait_id == trait_id:
                            continue
                        other_d_dir = dio.direction_dir(bundle, other_trait_id, model_id)
                        if not (other_d_dir / "standard.npy").exists():
                            continue
                        cross_results = await trait_judge.score_batch(
                            condition_responses, other_trait, concurrency=concurrency,
                        )
                        for r, xr in zip(condition_responses, cross_results, strict=True):
                            r.setdefault("cross_trait_scores", {})[other_trait_id] = xr.score

                score_time = time.time() - t_score
                logger.info(
                    "  alpha=%.1f: scored %d responses (%.1fs) | "
                    "progress: %d/%d alphas done",
                    alpha, n_responses, score_time,
                    alpha_idx + 1, len(alpha_values),
                )
            else:
                for r in condition_responses:
                    r["trait_score"] = None
                    r["coherence_score"] = None

            all_responses.extend(condition_responses)

        # Save per-response results
        out_dir = dio.resolve_paths(bundle)["report_dir"] / "phase3" / trait_id / model_id
        dio.save_responses_json(
            out_dir / f"steering_{direction_type}_responses.json",
            all_responses,
        )

        # Compute and save aggregates
        aggregates = _compute_aggregates(all_responses, alpha_values)
        dio.save_results_json(
            out_dir / f"steering_{direction_type}_aggregates.json",
            aggregates,
        )

        elapsed = time.time() - t0
        logger.info(
            "Steering eval complete: %s/%s direction=%s | "
            "%d total responses, %.1f minutes",
            trait_id, model_id, direction_type,
            len(all_responses), elapsed / 60,
        )

        return {
            "status": "completed",
            "direction_type": direction_type,
            "n_responses": len(all_responses),
            "n_alphas": len(alpha_values),
            "elapsed_seconds": round(elapsed, 1),
        }


def _compute_aggregates(
    responses: list[dict[str, Any]],
    alpha_values: list[float],
) -> dict[str, Any]:
    per_alpha = {}
    for alpha in alpha_values:
        alpha_resps = [r for r in responses if r["alpha"] == alpha]
        if not alpha_resps:
            continue

        trait_scores = [
            r["trait_score"] for r in alpha_resps
            if r.get("trait_score") is not None
        ]
        coherence_scores = [
            r["coherence_score"] for r in alpha_resps
            if r.get("coherence_score") is not None
        ]

        agg: dict[str, Any] = {
            "n_responses": len(alpha_resps),
            "trait_score_mean": float(np.mean(trait_scores)) if trait_scores else None,
            "trait_score_std": float(np.std(trait_scores)) if trait_scores else None,
            "coherence_score_mean": float(np.mean(coherence_scores)) if coherence_scores else None,
            "coherence_score_std": float(np.std(coherence_scores)) if coherence_scores else None,
        }

        # Cross-trait leakage
        cross_keys: set[str] = set()
        for r in alpha_resps:
            cross_keys.update(r.get("cross_trait_scores", {}).keys())

        cross_trait_means = {}
        for ct in sorted(cross_keys):
            ct_scores = [
                r["cross_trait_scores"][ct]
                for r in alpha_resps
                if r.get("cross_trait_scores", {}).get(ct) is not None
            ]
            if ct_scores:
                cross_trait_means[ct] = {
                    "mean": float(np.mean(ct_scores)),
                    "std": float(np.std(ct_scores)),
                }
        agg["cross_trait_leakage"] = cross_trait_means

        per_alpha[str(alpha)] = agg

    return {"per_alpha": per_alpha}

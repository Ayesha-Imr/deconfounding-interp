"""Pipeline stage: generate model responses, score them, filter, and extract activations."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.auroc_sweep import auroc_probe_sweep
from deconfounding_interp.backends import create_backend
from deconfounding_interp.llm.base import create_client
from deconfounding_interp.llm.judge import TraitJudge
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)


class RolloutsStage:
    name = "rollouts_and_activations"

    def __init__(self):
        self._backend = None
        self._loaded_model_id: str | None = None

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info(
                "[DRY RUN] Would generate rollouts for model=%s trait=%s variant=%s",
                job["model_id"], job["trait_id"], job["payload"].get("variant_index"),
            )
            return {"status": "dry_run"}
        return asyncio.run(self._run_async(job, context))

    async def _run_async(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        bundle = context.bundle
        model_id = job["model_id"]
        trait_id = job["trait_id"]
        payload = job["payload"]
        variant_index = payload["variant_index"]
        variant_kind = payload.get("variant_kind", "original")

        model_config = bundle.models[model_id]
        trait = bundle.traits[trait_id]
        gen_cfg = bundle.experiment.generation
        scoring_cfg = bundle.experiment.scoring
        extraction_cfg = bundle.experiment.extraction

        # Load assets
        assets = dio.load_results_json(dio.trait_raw_dir(bundle, trait_id) / "assets.json")

        # Pick the system prompt pair for this variant
        pos_prompt, neg_prompt = _select_prompt_pair(assets, variant_index, variant_kind)

        # Initialize backend (reload if model changed)
        backend_name = model_config.backend or bundle.experiment.backend
        if self._backend is None or self._loaded_model_id != model_id:
            if self._backend is not None:
                self._backend.unload_model()
            self._backend = create_backend(backend_name)
            self._backend.load_model(model_config)
            self._loaded_model_id = model_id

        # Generate rollouts
        questions = assets["extraction_questions"]
        rollouts_per = gen_cfg.get("rollouts_per_prompt_question", 5)
        temperature = gen_cfg.get("temperature", 1.0)
        top_p = gen_cfg.get("top_p", 0.95)
        max_new_tokens = gen_cfg.get("max_new_tokens", 256)

        all_responses = []
        for side, sys_prompt in [("pos", pos_prompt), ("neg", neg_prompt)]:
            for q in questions:
                prompts = [{"system_prompt": sys_prompt, "question": q}] * rollouts_per
                responses = self._backend.generate_responses(
                    prompts,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
                for resp in responses:
                    all_responses.append({
                        "system_prompt": sys_prompt,
                        "question": q,
                        "response": resp,
                        "side": side,
                        "variant_index": variant_index,
                        "variant_kind": variant_kind,
                    })

        # Score responses when requested. Pilot runs can use ``mode: none`` to
        # validate generation/extraction without requiring an external judge;
        # all responses are retained and used for the geometric smoke test.
        score_mode = scoring_cfg.get("mode", "judge")
        if score_mode == "none":
            logger.info(
                "Skipping judge for pilot: retaining all %d responses",
                len(all_responses),
            )
            for response in all_responses:
                response["score"] = None
            filtered_pos = [r for r in all_responses if r["side"] == "pos"]
            filtered_neg = [r for r in all_responses if r["side"] == "neg"]
        elif score_mode == "judge":
            llm_cfg = bundle.experiment.llm
            audit_dir = (
                dio.trait_interim_dir(bundle, trait_id, model_id)
                if llm_cfg.get("audit_csv") else None
            )
            judge_client = create_client(
                provider=llm_cfg.get("provider", "openai"),
                model=llm_cfg.get(
                    "judge_model",
                    bundle.judge.get("model", "gpt-4.1-mini-2025-04-14"),
                ),
                audit_dir=audit_dir,
            )
            judge = TraitJudge(judge_client, bundle.judge)
            concurrency = llm_cfg.get("judge_concurrency", 50)

            logger.info(
                "Scoring %d responses for trait=%s variant=%d",
                len(all_responses), trait_id, variant_index,
            )
            results = await judge.score_batch(
                all_responses, trait, concurrency=concurrency,
            )

            for resp_dict, judge_result in zip(
                all_responses, results, strict=True,
            ):
                resp_dict["score"] = judge_result.score

            min_pos = scoring_cfg.get("keep_positive_min_score", 50)
            max_neg = scoring_cfg.get("keep_negative_max_score", 50)
            filtered_pos = [
                r for r in all_responses
                if r["side"] == "pos" and r["score"] is not None
                and r["score"] > min_pos
            ]
            filtered_neg = [
                r for r in all_responses
                if r["side"] == "neg" and r["score"] is not None
                and r["score"] < max_neg
            ]
        else:
            raise ValueError(f"Unknown scoring.mode: {score_mode!r}")

        logger.info(
            "Filtered: %d pos (of %d), %d neg (of %d)",
            len(filtered_pos), sum(1 for r in all_responses if r["side"] == "pos"),
            len(filtered_neg), sum(1 for r in all_responses if r["side"] == "neg"),
        )

        # Save responses
        out_dir = dio.trait_interim_dir(bundle, trait_id, model_id)
        resp_path = out_dir / "responses" / f"variant_{variant_index:02d}.json"
        dio.save_responses_json(resp_path, all_responses)

        csv_path = out_dir / "responses" / f"variant_{variant_index:02d}.csv"
        _save_responses_csv(csv_path, all_responses)

        # Extract activations
        position = extraction_cfg.get("activation_position", "response_average")
        act_dir = out_dir / "activations" / f"variant_{variant_index:02d}"

        layer_acts = self._extract_side_activations(
            filtered_pos, filtered_neg, model_config, position,
        )
        dio.save_activations(act_dir, layer_acts)

        # AUROC sweep if no pre-selected layer
        selected_layer = payload.get("selected_layer")
        if selected_layer is None:
            pos_by_layer = {
                li: sides["pos"] for li, sides in layer_acts.items()
                if "pos" in sides
            }
            neg_by_layer = {
                li: sides["neg"] for li, sides in layer_acts.items()
                if "neg" in sides
            }
            if pos_by_layer and neg_by_layer:
                min_layer = model_config.num_layers // 5
                sweep = auroc_probe_sweep(pos_by_layer, neg_by_layer, min_layer=min_layer)
                dio.save_results_json(
                    out_dir / "selected_layer.json",
                    {
                        "best_layer": sweep.best_layer,
                        "best_auroc": sweep.best_auroc,
                        "all_aurocs": sweep.all_aurocs,
                    },
                )
                logger.info(
                    "AUROC sweep: best_layer=%d (auroc=%.3f)",
                    sweep.best_layer, sweep.best_auroc,
                )

        return {
            "status": "completed",
            "variant_index": variant_index,
            "n_pos": len(filtered_pos),
            "n_neg": len(filtered_neg),
        }

    def _extract_side_activations(
        self, filtered_pos, filtered_neg, model_config, position,
    ) -> dict[int, dict[str, np.ndarray]]:
        result: dict[int, dict[str, np.ndarray]] = {}

        for side, filtered in [("pos", filtered_pos), ("neg", filtered_neg)]:
            if not filtered:
                continue
            texts = []
            prompt_lengths = []
            for r in filtered:
                full_text = self._backend.format_chat(
                    r["system_prompt"], r["question"], r["response"],
                )
                texts.append(full_text)
                plen = self._backend.get_prompt_token_length(
                    r["system_prompt"], r["question"],
                )
                prompt_lengths.append(plen)

            acts_by_layer = self._backend.extract_activations(
                texts, prompt_lengths, position=position,
            )
            for layer_idx, arr in acts_by_layer.items():
                result.setdefault(layer_idx, {})[side] = arr

        return result


def _select_prompt_pair(
    assets: dict, variant_index: int, variant_kind: str,
) -> tuple[str, str]:
    if variant_kind == "original":
        pos = assets["positive_system_prompts"][variant_index]
        neg = assets["negative_system_prompts"][variant_index]
    else:
        n_originals = len(assets["positive_system_prompts"])
        idx = variant_index - n_originals
        pos = assets["positive_paraphrases"][idx]
        neg = assets["negative_paraphrases"][idx]
    return pos, neg


def _save_responses_csv(path, responses):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not responses:
        return
    fieldnames = list(responses[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(responses)

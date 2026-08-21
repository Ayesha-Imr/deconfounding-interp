from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deconfounding_interp.config import ConfigBundle


@dataclass(frozen=True)
class ManifestJob:
    phase: str
    model_id: str | None
    trait_id: str | None
    job_id: str
    payload: dict[str, Any]


def _int_setting(settings: dict[str, Any], key: str, default: int) -> int:
    return int(settings.get(key, default))


def build_manifest(bundle: ConfigBundle) -> dict[str, Any]:
    """Expand an experiment config into deterministic, shardable jobs."""

    generation = bundle.experiment.generation
    corrections = bundle.experiment.corrections
    n_pairs = _int_setting(generation, "system_prompt_pairs", 5)
    paraphrases_per_prompt = _int_setting(generation, "paraphrases_per_prompt", 1)
    variant_count = n_pairs * (1 + paraphrases_per_prompt)
    direction_types = tuple(
        corrections.get("direction_types", ["standard", "averaged", "subtracted"])
    )

    jobs: list[ManifestJob] = []
    for trait_id in bundle.traits:
        jobs.append(
            ManifestJob(
                phase="prompt_assets",
                model_id=None,
                trait_id=trait_id,
                job_id=f"prompt_assets__{trait_id}",
                payload={
                    "system_prompt_pairs": n_pairs,
                    "paraphrases_per_prompt": paraphrases_per_prompt,
                    "extraction_questions": _int_setting(generation, "extraction_questions", 20),
                    "evaluation_questions": _int_setting(generation, "evaluation_questions", 20),
                },
            )
        )

    for model_id, model in bundle.models.items():
        for trait_id in bundle.traits:
            selected_layer = model.trait_layers.get(trait_id)
            for variant_index in range(variant_count):
                variant_kind = "original" if variant_index < n_pairs else "paraphrase"
                jobs.append(
                    ManifestJob(
                        phase="rollouts_and_activations",
                        model_id=model_id,
                        trait_id=trait_id,
                        job_id=f"activations__{model_id}__{trait_id}__v{variant_index:02d}",
                        payload={
                            "variant_index": variant_index,
                            "variant_kind": variant_kind,
                            "rollouts_per_prompt_question": _int_setting(
                                generation, "rollouts_per_prompt_question", 5
                            ),
                            "selected_layer": selected_layer,
                        },
                    )
                )

            jobs.append(
                ManifestJob(
                    phase="direction_analysis",
                    model_id=model_id,
                    trait_id=trait_id,
                    job_id=f"analysis__{model_id}__{trait_id}",
                    payload={
                        "variant_count": variant_count,
                        "direction_types": list(direction_types),
                        "selected_layer": selected_layer,
                    },
                )
            )

            jobs.append(
                ManifestJob(
                    phase="null_analysis",
                    model_id=model_id,
                    trait_id=trait_id,
                    job_id=f"nulls__{model_id}__{trait_id}",
                    payload={
                        "variant_count": variant_count,
                        "selected_layer": selected_layer,
                        "repeats": int(
                            bundle.experiment.analysis.get("nulls", {}).get(
                                "repeats", 90,
                            )
                        ),
                    },
                )
            )

            for direction_type in direction_types:
                jobs.append(
                    ManifestJob(
                        phase="downstream_evaluation",
                        model_id=model_id,
                        trait_id=trait_id,
                        job_id=f"eval__{model_id}__{trait_id}__{direction_type}",
                        payload={
                            "direction_type": direction_type,
                            "alpha_values": bundle.experiment.steering.get("alpha_values", []),
                            "selected_layer": selected_layer,
                        },
                    )
                )

            layer_robustness_cfg = bundle.experiment.analysis.get(
                "layer_robustness", {}
            )
            if layer_robustness_cfg.get("enabled", False):
                holdout_index = int(
                    layer_robustness_cfg.get("holdout_index", variant_count - 1)
                )
                jobs.append(
                    ManifestJob(
                        phase="layer_robustness",
                        model_id=model_id,
                        trait_id=trait_id,
                        job_id=f"layer_robustness__{model_id}__{trait_id}",
                        payload={
                            "variant_count": variant_count,
                            "holdout_index": holdout_index,
                            "train_variant_indices": [
                                index for index in range(variant_count)
                                if index != holdout_index
                            ],
                            "direction_types": ["standard", "random", "sign_reversed"],
                        },
                    )
                )

            position_cfg = bundle.experiment.analysis.get(
                "position_robustness", {}
            )
            if position_cfg.get("enabled", False):
                positions = [str(position) for position in position_cfg.get("positions", [])]
                source_interim_root = str(
                    position_cfg.get("source_interim_dir", "data/interim")
                )
                output_interim_root = str(
                    position_cfg.get("output_interim_dir", "data/interim/positions")
                )
                holdout_index = int(
                    position_cfg.get("holdout_index", variant_count - 1)
                )
                jobs.append(
                    ManifestJob(
                        phase="position_reextraction",
                        model_id=model_id,
                        trait_id=trait_id,
                        job_id=f"position_reextract__{model_id}__{trait_id}",
                        payload={
                            "variant_count": variant_count,
                            "positions": positions,
                            "source_interim_root": source_interim_root,
                            "output_interim_root": output_interim_root,
                        },
                    )
                )
                for position in positions:
                    jobs.append(
                        ManifestJob(
                            phase="position_layer_robustness",
                            model_id=model_id,
                            trait_id=trait_id,
                            job_id=(
                                f"position_layer_robustness__{model_id}__"
                                f"{trait_id}__{position}"
                            ),
                            payload={
                                "variant_count": variant_count,
                                "holdout_index": holdout_index,
                                "train_variant_indices": [
                                    index for index in range(variant_count)
                                    if index != holdout_index
                                ],
                                "position": position,
                                "interim_root": output_interim_root,
                                "direction_types": [
                                    "standard", "random", "sign_reversed",
                                ],
                            },
                        )
                    )

            jobs.append(
                ManifestJob(
                    phase="probing",
                    model_id=model_id,
                    trait_id=trait_id,
                    job_id=f"probing__{model_id}__{trait_id}",
                    payload={
                        "direction_types": list(direction_types),
                        "variant_count": variant_count,
                        "selected_layer": selected_layer,
                    },
                )
            )

    jobs.append(
        ManifestJob(
            phase="phase3_summary",
            model_id=None,
            trait_id=None,
            job_id="phase3_summary",
            payload={"write_figures": True},
        )
    )

    jobs.append(
        ManifestJob(
            phase="direction_summary",
            model_id=None,
            trait_id=None,
            job_id="direction_summary",
            payload={
                "variant_count": variant_count,
                "random_seed": bundle.experiment.random_seed,
                "random_baseline_n": 100,
                "write_figures": True,
            },
        )
    )

    return {
        "experiment_id": bundle.experiment.id,
        "random_seed": bundle.experiment.random_seed,
        "model_ids": list(bundle.models),
        "trait_ids": list(bundle.traits),
        "variant_count_per_trait_model": variant_count,
        "jobs": [asdict(job) for job in jobs],
    }


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path

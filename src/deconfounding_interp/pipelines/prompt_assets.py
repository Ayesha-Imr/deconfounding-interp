"""Pipeline stage: generate trait system prompts (LLM) and load questions from dataset."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from deconfounding_interp import io as dio
from deconfounding_interp.llm.base import LLMClient, create_client
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_QUESTIONS_PATH = Path("data/questions.json")


class PromptAssetsStage:
    name = "prompt_assets"

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info("[DRY RUN] Would generate prompt assets for trait=%s", job["trait_id"])
            return {"status": "dry_run"}
        return asyncio.run(self._run_async(job, context))

    async def _run_async(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        bundle = context.bundle
        trait = bundle.traits[job["trait_id"]]
        llm_cfg = bundle.experiment.llm
        payload = job.get("payload", {})

        prompt_mode = llm_cfg.get("prompt_asset_mode", "llm")
        n_paraphrases = int(payload.get("paraphrases_per_prompt", 1))
        client = None
        if prompt_mode != "deterministic" or n_paraphrases:
            audit_dir = (
                dio.trait_raw_dir(bundle, trait.id) if llm_cfg.get("audit_csv") else None
            )
            client = create_client(
                provider=llm_cfg.get("provider", "openai"),
                model=llm_cfg.get("generation_model", "gpt-4.1-mini-2025-04-14"),
                audit_dir=audit_dir,
            )

        if prompt_mode == "deterministic":
            assets = self._deterministic_system_prompts(trait, payload)
        else:
            assert client is not None
            assets = await self._generate_system_prompts(client, trait, payload, llm_cfg)
        assets = self._load_questions(assets, bundle, payload)
        if n_paraphrases:
            assert client is not None
            paraphrases = await self._generate_paraphrases(
                client, assets, llm_cfg, n_paraphrases=n_paraphrases,
            )
        else:
            paraphrases = {"positive": [], "negative": []}
        assets["positive_paraphrases"] = paraphrases["positive"]
        assets["negative_paraphrases"] = paraphrases["negative"]

        out_dir = dio.trait_raw_dir(bundle, trait.id)
        dio.save_results_json(out_dir / "assets.json", assets)
        logger.info("Saved prompt assets to %s", out_dir / "assets.json")

        return {"status": "completed", "trait_id": trait.id, "output": str(out_dir / "assets.json")}

    @staticmethod
    def _deterministic_system_prompts(trait, payload: dict[str, Any]) -> dict[str, Any]:
        n_pairs = int(payload.get("system_prompt_pairs", 1))
        return {
            "positive_system_prompts": [
                (
                    f"{trait.prompt_generation['positive_instruction_seed']} "
                    f"Use controlled template wording {index + 1}."
                )
                for index in range(n_pairs)
            ],
            "negative_system_prompts": [
                (
                    f"{trait.prompt_generation['negative_instruction_seed']} "
                    f"Use controlled template wording {index + 1}."
                )
                for index in range(n_pairs)
            ],
        }

    async def _generate_system_prompts(
        self, client: LLMClient, trait, payload: dict, llm_cfg: dict,
    ) -> dict[str, Any]:
        template = (_PROMPTS_DIR / "generate_trait_assets.txt").read_text()
        prompt = template.format(
            trait_display_name=trait.display_name,
            positive_definition=trait.positive_definition,
            negative_definition=trait.negative_definition,
            n_prompt_pairs=payload.get("system_prompt_pairs", 5),
        )
        resp = await client.generate(
            [{"role": "user", "content": prompt}],
            temperature=llm_cfg.get("generation_temperature", 0.7),
            max_tokens=llm_cfg.get("generation_max_tokens", 4096),
        )
        assets = _parse_json_response(resp.content)
        missing = {"positive_system_prompts", "negative_system_prompts"} - set(assets.keys())
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")
        return assets

    @staticmethod
    def _load_questions(assets: dict, bundle, payload: dict[str, Any]) -> dict:
        path = bundle.project_root / _QUESTIONS_PATH
        if not path.exists():
            raise RuntimeError(
                f"Questions file not found at {path}. "
                "Run `deconfound sample-questions` first."
            )
        questions = json.loads(path.read_text())
        n_extract = payload.get("extraction_questions")
        n_eval = payload.get("evaluation_questions")
        assets["extraction_questions"] = questions["extraction_questions"][:n_extract]
        assets["evaluation_questions"] = questions["evaluation_questions"][:n_eval]
        return assets

    async def _generate_paraphrases(
        self,
        client: LLMClient,
        assets: dict,
        llm_cfg: dict,
        *,
        n_paraphrases: int = 1,
    ) -> dict[str, list[str]]:
        template = (_PROMPTS_DIR / "paraphrase_system_prompt.txt").read_text()
        result: dict[str, list[str]] = {"positive": [], "negative": []}

        for side in ("positive", "negative"):
            key = f"{side}_system_prompts"
            for prompt_text in assets[key][:n_paraphrases]:
                formatted = template.format(system_prompt=prompt_text)
                resp = await client.generate(
                    [{"role": "user", "content": formatted}],
                    temperature=llm_cfg.get("generation_temperature", 0.7),
                    max_tokens=1024,
                )
                result[side].append(resp.content.strip())

        return result


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response that may contain markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # skip ```json
        end = next((i for i, ln in enumerate(lines) if ln.strip() == "```"), len(lines))
        cleaned = "\n".join(lines[:end])
    return json.loads(cleaned)

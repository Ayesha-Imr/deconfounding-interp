"""Pipeline stage: generate trait prompts, questions, and paraphrases via LLM."""

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

        audit_dir = dio.trait_raw_dir(bundle, trait.id) if llm_cfg.get("audit_csv") else None
        client = create_client(
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg.get("generation_model", "gpt-4.1-mini-2025-04-14"),
            audit_dir=audit_dir,
        )

        assets = await self._generate_assets(client, trait, payload, llm_cfg)
        paraphrases = await self._generate_paraphrases(client, assets, llm_cfg)
        assets["positive_paraphrases"] = paraphrases["positive"]
        assets["negative_paraphrases"] = paraphrases["negative"]

        out_dir = dio.trait_raw_dir(bundle, trait.id)
        dio.save_results_json(out_dir / "assets.json", assets)
        logger.info("Saved prompt assets to %s", out_dir / "assets.json")

        return {"status": "completed", "trait_id": trait.id, "output": str(out_dir / "assets.json")}

    async def _generate_assets(
        self, client: LLMClient, trait, payload: dict, llm_cfg: dict,
    ) -> dict[str, Any]:
        template = (_PROMPTS_DIR / "generate_trait_assets.txt").read_text()
        prompt = template.format(
            trait_display_name=trait.display_name,
            positive_definition=trait.positive_definition,
            negative_definition=trait.negative_definition,
            n_prompt_pairs=payload.get("system_prompt_pairs", 5),
            n_extraction_questions=payload.get("extraction_questions", 20),
            n_evaluation_questions=payload.get("evaluation_questions", 20),
        )
        resp = await client.generate(
            [{"role": "user", "content": prompt}],
            temperature=llm_cfg.get("generation_temperature", 0.7),
            max_tokens=llm_cfg.get("generation_max_tokens", 4096),
        )
        assets = _parse_json_response(resp.content)
        expected_keys = {
            "positive_system_prompts", "negative_system_prompts",
            "extraction_questions", "evaluation_questions",
        }
        missing = expected_keys - set(assets.keys())
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")
        return assets

    async def _generate_paraphrases(
        self, client: LLMClient, assets: dict, llm_cfg: dict,
    ) -> dict[str, list[str]]:
        template = (_PROMPTS_DIR / "paraphrase_system_prompt.txt").read_text()
        result: dict[str, list[str]] = {"positive": [], "negative": []}

        for side in ("positive", "negative"):
            key = f"{side}_system_prompts"
            for prompt_text in assets[key]:
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

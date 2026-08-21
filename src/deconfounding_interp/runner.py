"""Pipeline runner: dispatches manifest jobs to stages with checkpoint/resume."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from deconfounding_interp.config import ConfigBundle
from deconfounding_interp.pipelines.base import StageContext
from deconfounding_interp.pipelines.direction_analysis import DirectionAnalysisStage
from deconfounding_interp.pipelines.direction_summary import DirectionSummaryStage
from deconfounding_interp.pipelines.downstream_evaluation import DownstreamEvaluationStage
from deconfounding_interp.pipelines.layer_robustness import LayerRobustnessStage
from deconfounding_interp.pipelines.null_analysis import NullAnalysisStage
from deconfounding_interp.pipelines.phase3_summary import Phase3SummaryStage
from deconfounding_interp.pipelines.probing import ProbingStage
from deconfounding_interp.pipelines.prompt_assets import PromptAssetsStage
from deconfounding_interp.pipelines.rollouts import RolloutsStage
from deconfounding_interp.provenance import write_run_metadata

logger = logging.getLogger(__name__)

STAGE_REGISTRY: dict[str, type] = {
    "prompt_assets": PromptAssetsStage,
    "rollouts_and_activations": RolloutsStage,
    "direction_analysis": DirectionAnalysisStage,
    "direction_summary": DirectionSummaryStage,
    "downstream_evaluation": DownstreamEvaluationStage,
    "probing": ProbingStage,
    "phase3_summary": Phase3SummaryStage,
    "null_analysis": NullAnalysisStage,
    "layer_robustness": LayerRobustnessStage,
}


class Checkpoint:
    """Tracks completed job IDs for resume support."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = self._load()

    def is_completed(self, job_id: str) -> bool:
        return job_id in self.data.get("completed", {})

    def mark_completed(self, job_id: str, result: dict[str, Any]) -> None:
        self.data.setdefault("completed", {})[job_id] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "result": result,
        }
        self._save()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)


class PipelineRunner:

    def __init__(
        self,
        bundle: ConfigBundle,
        run_dir: Path,
        dry_run: bool = False,
    ):
        self.context = StageContext(bundle=bundle, run_dir=run_dir, dry_run=dry_run)
        self.checkpoint = Checkpoint(run_dir / "checkpoint.json")
        self._stage_cache: dict[str, Any] = {}

    def _get_stage(self, phase: str):
        if phase not in STAGE_REGISTRY:
            return None
        if phase not in self._stage_cache:
            self._stage_cache[phase] = STAGE_REGISTRY[phase]()
        return self._stage_cache[phase]

    def run_manifest(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        write_run_metadata(self.context.bundle, manifest, self.context.run_dir)
        jobs = manifest["jobs"]
        results = []
        total = len(jobs)
        for i, job in enumerate(jobs, 1):
            job_id = job["job_id"]
            if self.checkpoint.is_completed(job_id):
                logger.info("[%d/%d] Skipping completed job %s", i, total, job_id)
                continue
            logger.info("[%d/%d] Running job %s (phase=%s)", i, total, job_id, job["phase"])
            result = self._run_single(job)
            results.append(result)
        return results

    def run_phase(self, manifest: dict[str, Any], phase: str) -> list[dict[str, Any]]:
        write_run_metadata(self.context.bundle, manifest, self.context.run_dir)
        jobs = [j for j in manifest["jobs"] if j["phase"] == phase]
        if not jobs:
            logger.warning("No jobs found for phase=%s", phase)
            return []
        results = []
        for i, job in enumerate(jobs, 1):
            job_id = job["job_id"]
            if self.checkpoint.is_completed(job_id):
                logger.info("[%d/%d] Skipping completed job %s", i, len(jobs), job_id)
                continue
            logger.info("[%d/%d] Running job %s", i, len(jobs), job_id)
            result = self._run_single(job)
            results.append(result)
        return results

    def run_job(self, manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
        write_run_metadata(self.context.bundle, manifest, self.context.run_dir)
        job = next((j for j in manifest["jobs"] if j["job_id"] == job_id), None)
        if job is None:
            raise ValueError(f"Job {job_id!r} not found in manifest")
        return self._run_single(job)

    def _run_single(self, job: dict[str, Any]) -> dict[str, Any]:
        stage = self._get_stage(job["phase"])
        if stage is None:
            logger.warning(
                "Skipping job %s: phase %r not implemented",
                job["job_id"], job["phase"],
            )
            return {"status": "skipped", "reason": f"phase {job['phase']!r} not implemented"}
        t0 = time.monotonic()
        result = stage.run(job, self.context)
        elapsed = time.monotonic() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        if not self.context.dry_run and result.get("status") not in {"blocked", "failed"}:
            self.checkpoint.mark_completed(job["job_id"], result)
        logger.info("Job %s completed in %.1fs", job["job_id"], elapsed)
        return result

"""Pipeline stage: aggregate direction-analysis outputs."""

from __future__ import annotations

import logging
from typing import Any

from deconfounding_interp.analysis.direction_reports import (
    build_direction_summary,
    configured_variant_count,
)
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)


class DirectionSummaryStage:
    name = "direction_summary"

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        bundle = context.bundle
        payload = job.get("payload", {})
        variant_count = int(payload.get("variant_count", configured_variant_count(bundle)))
        random_seed = int(payload.get("random_seed", bundle.experiment.random_seed))
        random_baseline_n = int(payload.get("random_baseline_n", 100))

        if context.dry_run:
            logger.info(
                "[DRY RUN] Would build direction summary for %d variants per trait/model",
                variant_count,
            )
            return {"status": "dry_run"}

        return build_direction_summary(
            bundle,
            variant_count=variant_count,
            random_seed=random_seed,
            random_baseline_n=random_baseline_n,
            write_figures=bool(payload.get("write_figures", True)),
        )

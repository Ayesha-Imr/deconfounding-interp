from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from deconfounding_interp.config import ConfigBundle


@dataclass(frozen=True)
class StageContext:
    bundle: ConfigBundle
    run_dir: Path
    dry_run: bool = False


class PipelineStage(Protocol):
    name: str

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        """Run one manifest job and return serializable metadata."""


class NotImplementedStage:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        raise NotImplementedError(
            f"Pipeline stage `{self.name}` is a placeholder. "
            "Implement a backend in src/deconfounding_interp/pipelines/."
        )

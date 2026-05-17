from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or inconsistent."""


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a JSON, TOML, or YAML mapping from disk."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(config_path.read_text())
    elif suffix == ".toml":
        data = tomllib.loads(config_path.read_text())
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ConfigError(
                "YAML configs require PyYAML. Install with `pip install -e .`."
            ) from exc
        data = yaml.safe_load(config_path.read_text()) or {}
    else:
        raise ConfigError(f"Unsupported config extension for {config_path}")

    if not isinstance(data, dict):
        raise ConfigError(f"Config must contain a mapping at top level: {config_path}")
    return data


def _required_str(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required string `{key}` in {source}")
    return value


def _string_list(data: dict[str, Any], key: str, source: Path) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"`{key}` must be a non-empty list in {source}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ConfigError(f"`{key}` must contain only non-empty strings in {source}")
    return tuple(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


@dataclass(frozen=True)
class TraitConfig:
    id: str
    display_name: str
    kind: str
    expected_surface_confound: str
    description: str
    polarity: dict[str, str]
    positive_definition: str
    negative_definition: str
    prompt_generation: dict[str, Any]
    judge: dict[str, Any]
    source: Path

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: str | Path) -> TraitConfig:
        path = Path(source)
        polarity = data.get("polarity", {})
        prompt_generation = data.get("prompt_generation", {})
        judge = data.get("judge", {})
        if not isinstance(polarity, dict):
            raise ConfigError(f"`polarity` must be a mapping in {path}")
        if not isinstance(prompt_generation, dict):
            raise ConfigError(f"`prompt_generation` must be a mapping in {path}")
        if not isinstance(judge, dict):
            raise ConfigError(f"`judge` must be a mapping in {path}")
        return cls(
            id=_required_str(data, "id", path),
            display_name=str(data.get("display_name") or data.get("id")),
            kind=_required_str(data, "kind", path),
            expected_surface_confound=str(data.get("expected_surface_confound", "unknown")),
            description=str(data.get("description", "")),
            polarity={str(key): str(value) for key, value in polarity.items()},
            positive_definition=_required_str(data, "positive_definition", path),
            negative_definition=_required_str(data, "negative_definition", path),
            prompt_generation=dict(prompt_generation),
            judge=dict(judge),
            source=path,
        )


@dataclass(frozen=True)
class ModelConfig:
    id: str
    display_name: str
    provider: str
    model_name: str
    tokenizer_name: str
    dtype: str
    device_map: str
    trust_remote_code: bool
    chat_template: str
    residual_stream_name: str
    num_layers: int
    default_max_new_tokens: int
    trait_layers: dict[str, int | None]
    backend: str | None
    source: Path

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: str | Path) -> ModelConfig:
        path = Path(source)
        trait_layers = data.get("trait_layers", {})
        if not isinstance(trait_layers, dict):
            raise ConfigError(f"`trait_layers` must be a mapping in {path}")
        raw_backend = data.get("backend")
        return cls(
            id=_required_str(data, "id", path),
            display_name=str(data.get("display_name") or data.get("id")),
            provider=str(data.get("provider", "huggingface")),
            model_name=_required_str(data, "model_name", path),
            tokenizer_name=str(data.get("tokenizer_name") or data.get("model_name")),
            dtype=str(data.get("dtype", "bfloat16")),
            device_map=str(data.get("device_map", "auto")),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            chat_template=str(data.get("chat_template", "tokenizer_default")),
            residual_stream_name=_required_str(data, "residual_stream_name", path),
            num_layers=int(data.get("num_layers", 0)),
            default_max_new_tokens=int(data.get("default_max_new_tokens", 256)),
            trait_layers={str(key): _optional_int(value) for key, value in trait_layers.items()},
            backend=str(raw_backend) if raw_backend is not None else None,
            source=path,
        )


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    description: str
    random_seed: int
    models: tuple[str, ...]
    traits: tuple[str, ...]
    judge: str | None
    paths: dict[str, Any]
    generation: dict[str, Any]
    scoring: dict[str, Any]
    extraction: dict[str, Any]
    surface_form: dict[str, Any]
    corrections: dict[str, Any]
    analysis: dict[str, Any]
    steering: dict[str, Any]
    backend: str
    llm: dict[str, Any]
    source: Path

    @classmethod
    def from_mapping(cls, data: dict[str, Any], source: str | Path) -> ExperimentConfig:
        path = Path(source)
        return cls(
            id=_required_str(data, "id", path),
            description=str(data.get("description", "")),
            random_seed=int(data.get("random_seed", 0)),
            models=_string_list(data, "models", path),
            traits=_string_list(data, "traits", path),
            judge=data.get("judge"),
            paths=dict(data.get("paths", {})),
            generation=dict(data.get("generation", {})),
            scoring=dict(data.get("scoring", {})),
            extraction=dict(data.get("extraction", {})),
            surface_form=dict(data.get("surface_form", {})),
            corrections=dict(data.get("corrections", {})),
            analysis=dict(data.get("analysis", {})),
            steering=dict(data.get("steering", {})),
            backend=str(data.get("backend", "huggingface")),
            llm=dict(data.get("llm", {})),
            source=path,
        )


@dataclass(frozen=True)
class ConfigBundle:
    project_root: Path
    experiment: ExperimentConfig
    models: dict[str, ModelConfig]
    traits: dict[str, TraitConfig]
    judge: dict[str, Any] | None

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self.models)

    @property
    def trait_ids(self) -> tuple[str, ...]:
        return tuple(self.traits)


def infer_project_root(experiment_path: str | Path) -> Path:
    path = Path(experiment_path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "configs").is_dir():
            return parent
    if path.parent.name == "experiments" and path.parent.parent.name == "configs":
        return path.parent.parent.parent
    return Path.cwd().resolve()


def _with_suffix_candidates(path: Path) -> tuple[Path, ...]:
    if path.suffix:
        return (path,)
    return (path, path.with_suffix(".yaml"), path.with_suffix(".yml"), path.with_suffix(".json"))


def resolve_config_ref(ref: str, project_root: Path, current_dir: Path, kind: str) -> Path:
    raw = Path(ref)
    bases = [raw] if raw.is_absolute() else [current_dir / raw, project_root / raw]
    if not raw.is_absolute():
        bases.extend([project_root / "configs" / raw, project_root / "configs" / kind / raw])

    for base in bases:
        for candidate in _with_suffix_candidates(base):
            if candidate.exists():
                return candidate.resolve()
    raise ConfigError(f"Could not resolve {kind} config reference `{ref}`")


def load_config_bundle(
    experiment_path: str | Path,
    project_root: str | Path | None = None,
) -> ConfigBundle:
    exp_path = Path(experiment_path).resolve()
    root = Path(project_root).resolve() if project_root else infer_project_root(exp_path)
    experiment = ExperimentConfig.from_mapping(load_mapping(exp_path), exp_path)

    models: dict[str, ModelConfig] = {}
    for ref in experiment.models:
        model_path = resolve_config_ref(ref, root, exp_path.parent, "models")
        model = ModelConfig.from_mapping(load_mapping(model_path), model_path)
        if model.id in models:
            raise ConfigError(f"Duplicate model id `{model.id}`")
        models[model.id] = model

    traits: dict[str, TraitConfig] = {}
    for ref in experiment.traits:
        trait_path = resolve_config_ref(ref, root, exp_path.parent, "traits")
        trait = TraitConfig.from_mapping(load_mapping(trait_path), trait_path)
        if trait.id in traits:
            raise ConfigError(f"Duplicate trait id `{trait.id}`")
        traits[trait.id] = trait

    judge = None
    if experiment.judge:
        judge_path = resolve_config_ref(experiment.judge, root, exp_path.parent, "judges")
        judge = load_mapping(judge_path)

    bundle = ConfigBundle(
        project_root=root,
        experiment=experiment,
        models=models,
        traits=traits,
        judge=judge,
    )
    errors = validate_bundle(bundle)
    if errors:
        raise ConfigError("\n".join(errors))
    return bundle


def validate_bundle(bundle: ConfigBundle) -> list[str]:
    errors: list[str] = []
    trait_ids = set(bundle.traits)

    if not bundle.models:
        errors.append("Experiment must include at least one model")
    if not bundle.traits:
        errors.append("Experiment must include at least one trait")

    for trait in bundle.traits.values():
        if "positive" not in trait.polarity or "negative" not in trait.polarity:
            errors.append(f"Trait `{trait.id}` must define polarity.positive and polarity.negative")
        for key in ("positive_instruction_seed", "negative_instruction_seed"):
            if not trait.prompt_generation.get(key):
                errors.append(f"Trait `{trait.id}` missing prompt_generation.{key}")

    valid_backends = {"huggingface", "vllm"}
    if bundle.experiment.backend not in valid_backends:
        errors.append(
            f"Experiment backend must be one of {valid_backends}, "
            f"got `{bundle.experiment.backend}`"
        )

    for model in bundle.models.values():
        if model.backend is not None and model.backend not in valid_backends:
            errors.append(
                f"Model `{model.id}` backend must be one of "
                f"{valid_backends}, got `{model.backend}`"
            )
        if model.num_layers <= 0:
            errors.append(f"Model `{model.id}` must define a positive num_layers")
        unknown_layers = sorted(set(model.trait_layers) - trait_ids)
        if unknown_layers:
            errors.append(
                f"Model `{model.id}` has trait_layers for unknown traits: "
                + ", ".join(unknown_layers)
            )
        for trait_id, layer in model.trait_layers.items():
            if layer is not None and not 0 <= layer < model.num_layers:
                errors.append(
                    f"Model `{model.id}` layer for `{trait_id}` is {layer}, "
                    f"outside [0, {model.num_layers})"
                )

    return errors

# Deconfounding Interp

Counterfactual baselines for contrastive activation extraction: measure how much a DiM concept direction is target concept versus surface-form wording artifact.

The repo is set up so new traits are plug-and-play. Add one config in `configs/traits/`, include it in an experiment YAML, and the manifest builder will enumerate the required prompt generation, activation extraction, analysis, and evaluation jobs.

## Layout

```text
configs/
  experiments/        Experiment-level knobs and trait/model lists
  models/             Hugging Face model configs and layer defaults
  traits/             One file per trait; this is the main extension point
  judges/             Rubric templates for trait scoring
src/deconfounding_interp/
  cli.py              Config validation and manifest generation
  config.py           Config loading and schema checks
  manifest.py         Expands traits x models x variants into jobs
  directions.py       DiM, cosine, averaging, and subspace correction math
  analysis/           CPU-only stability and surface-overlap analysis helpers
  pipelines/          Stage interfaces for generation/extraction/evaluation
  prompts/            Reusable LLM prompt templates
outputs/              Local manifests, directions, reports, ignored by git
```

## Quickstart

Preferred `uv` workflow:

```bash
uv sync --all-extras
uv run deconfound validate-config --config configs/experiments/main.yaml
uv run deconfound make-manifest --config configs/experiments/main.yaml --output outputs/manifests/main.json
```

Pip fallback:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,viz,api]'
deconfound validate-config --config configs/experiments/main.yaml
deconfound make-manifest --config configs/experiments/main.yaml --output outputs/manifests/main.json
```

Without installing the console script, use:

```bash
PYTHONPATH=src python -m deconfounding_interp.cli validate-config --config configs/experiments/main.yaml
```

Common development commands:

```bash
make validate
make manifest
make test
make lint
```

## Adding A Trait

1. Copy `configs/traits/sycophancy.yaml` to `configs/traits/<new_trait>.yaml`.
2. Change `id`, definitions, prompt seeds, and judge anchors.
3. Add the trait path to `configs/experiments/main.yaml` under `traits`.
4. Optionally add a published best layer in each `configs/models/*.yaml`; otherwise layer selection falls back to the configured AUROC probe sweep.
5. Run `uv run deconfound validate-config` and regenerate the manifest.

No Python changes should be needed for ordinary trait additions.

## Current Scope

This is a scaffold, not a full reproduction yet. It includes the config model, job manifest expansion, direction math, and analysis primitives. GPU-facing stages are intentionally represented as pipeline interfaces/stubs so model-specific generation, activation capture, scoring, and steering can be filled in without changing trait configs.

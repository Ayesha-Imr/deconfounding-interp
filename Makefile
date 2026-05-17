.PHONY: sync validate manifest test lint

sync:
	uv sync --all-extras

validate:
	uv run deconfound validate-config --config configs/experiments/main.yaml

manifest:
	uv run deconfound make-manifest --config configs/experiments/main.yaml --output outputs/manifests/main.json

test:
	uv run --extra dev pytest

lint:
	uv run --extra dev ruff check .

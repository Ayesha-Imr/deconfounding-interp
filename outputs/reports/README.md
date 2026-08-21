# Report artifact status

The files in this directory are the original exploratory Phase 2/3 snapshot.
They remain tracked for reproducibility of the historical figures, but they
predate the current provenance/checksum repairs and stochastic-null rerun. Do
not use them as submission-grade evidence without regenerating and auditing
from a current immutable manifest.

Current 8B work is intentionally isolated under private run-specific paths:

- `configs/experiments/geometry_stochastic_null_8b.yaml` — stochastic geometry
  and null checks for Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
- `configs/experiments/causal_controls_8b.yaml` — judge-free standard/random/
  sign-reversed steering-hook smoke using those artifacts.

The current runs retain null trait/coherence scores until an approved objective
scorer or judge provider is available. No behavioral claim should be inferred
from this legacy directory.

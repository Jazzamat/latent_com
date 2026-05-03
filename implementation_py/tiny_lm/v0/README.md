# Tiny LM v0

Single-agent, character-level decoder-only transformer baseline.

## Phase 1 goals

- Train a tiny autoregressive LM end to end
- Generate text autoregressively
- Expose hidden states for later latent communication experiments

## First steps

1. Put a text file at `data/input.txt`
2. Run `../.venv/bin/python train.py`
3. Run `../.venv/bin/python generate.py`

## Environment setup

1. From repository root (`tiny_lm/`), create a virtualenv: `python -m venv .venv`
2. Install dependencies from `v0`: `.venv/bin/python -m pip install -r v0/requirements.txt`

## Testing

- Fast unit tests (from `v0/`): `../.venv/bin/python -m pytest -q`
- Separate blackbox tests: `../.venv/bin/python -m pytest -q tests_blackbox`
- ToM qualitative run (prints responses): `../.venv/bin/python -m pytest -q tests_blackbox/test_tom_qualitative.py -s`
- Blackbox device policy: CPU by default, GPU opt-in with `TINYLM_BLACKBOX_USE_GPU=1`
  - Example: `TINYLM_BLACKBOX_USE_GPU=1 ../.venv/bin/python -m pytest -q tests_blackbox`

## Notes

- This version is character-level for simplicity
- Hidden states can be returned with `return_hidden_states=True`
- Blackbox tests are intentionally separate from fast unit tests
- Historical details and experiment notes are in `report.txt`

# Tiny LM Versions

This repository is versioned by research stage.

- `v0/`: single-agent baseline (character-level LM, generation, tests)
- `v1/`: latent communication stage (multi-agent sender/receiver setup)

## Quick start

1. From `tiny_lm/`, create env: `python -m venv .venv`
2. Install dependencies for a version:
   - v0: `.venv/bin/python -m pip install -r v0/requirements.txt`
   - v1: `.venv/bin/python -m pip install -r v1/requirements.txt`

## Run tests

- v0 unit tests: `.venv/bin/python -m pytest -q v0/tests`
- v0 blackbox tests: `.venv/bin/python -m pytest -q v0/tests_blackbox`
- v0 blackbox GPU opt-in (if available): `TINYLM_BLACKBOX_USE_GPU=1 .venv/bin/python -m pytest -q v0/tests_blackbox`
- v1 unit tests: `.venv/bin/python -m pytest -q v1/tests`

## v1 robustness sweep

- Multi-seed OOD robustness sweep: `.venv/bin/python v1/sweep_comm.py`
  - default uses seeds `42..51` with fixed held-out visibility signatures across seeds
- Message-dimension sweep (multi-seed): `.venv/bin/python v1/sweep_message_dim.py`

## Reports

Each version has a detailed `report.txt`:

- `v0/report.txt`
- `v1/report.txt`

Top-level project tracker:

- `../report.txt`

Use these reports as the primary onboarding notes when resuming work.

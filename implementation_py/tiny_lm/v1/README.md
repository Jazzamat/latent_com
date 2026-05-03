# Tiny LM v1

v1 is the latent-communication branch.

Focus:
- Sender and receiver agents communicate through continuous latent vectors.
- Target task is belief-state prediction in synthetic false-belief episodes.
- English generation quality is not the primary objective.
- Evaluation includes IID and OOD ablations via visibility-signature holdout.

## Key files

- `comm_data.py`: synthetic belief-task episode generation and dataset builder
- `comm_model.py`: sender/receiver model with a continuous message bottleneck
- `train_comm.py`: training loop and ablation-aware evaluation
- `eval_comm.py`: evaluation helpers for message ablations
- `sweep_comm.py`: multi-seed robustness sweep with aggregate stats
- `sweep_message_dim.py`: message-dimension sweep built on multi-seed runs
- `tests/`: unit tests for data and communication model contracts
- `report.txt`: research log, aims, process, and results

## Setup

From repository root (`tiny_lm/`):

1. `.venv/bin/python -m pip install -r v1/requirements.txt`
2. `.venv/bin/python -m pytest -q v1/tests`

## Train baseline communication run

From repository root (`tiny_lm/`):

- `.venv/bin/python v1/train_comm.py`

## Run multi-seed robustness sweep

From repository root (`tiny_lm/`):

- `.venv/bin/python v1/sweep_comm.py`

Defaults:

- Seeds: `42..51` (10 runs)
- Held-out visibility signatures are frozen across seeds for fairer robustness comparison.

Useful options:

- Short test run: `.venv/bin/python v1/sweep_comm.py --seeds 42 43 44 --epochs 10`
- Explicit holdout-signature seed: `.venv/bin/python v1/sweep_comm.py --ood-signature-seed 42`

## Run message-dimension sweep

From repository root (`tiny_lm/`):

- `.venv/bin/python v1/sweep_message_dim.py`

Useful options:

- Faster pilot: `.venv/bin/python v1/sweep_message_dim.py --message-dims 8 16 32 --seeds 42 43 44 --epochs 10`

The training script reports both IID and OOD metrics:

- `iid_with_message_acc`, `iid_zero_message_acc`, `iid_shuffled_message_acc`, `iid_comm_gain`
- `ood_with_message_acc`, `ood_zero_message_acc`, `ood_shuffled_message_acc`, `ood_comm_gain`

Terminology:

- IID = in-distribution evaluation (same pattern family as train)
- OOD = out-of-distribution evaluation (held-out visibility-signature patterns)

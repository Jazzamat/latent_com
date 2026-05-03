import pytest

from sweep_message_dim import run_message_dim_sweep
from train_comm import CommTrainConfig


def test_run_message_dim_sweep_requires_nonempty_dims() -> None:
    with pytest.raises(ValueError):
        run_message_dim_sweep(base_config=CommTrainConfig(), message_dims=[], seeds=[42])


def test_run_message_dim_sweep_builds_per_dim_summary(monkeypatch) -> None:
    calls = []

    def fake_run_seed_sweep(base_config, seeds):
        calls.append((base_config.message_dim, list(seeds)))
        dim = base_config.message_dim
        return {
            "ood_signature_seed": 42,
            "aggregate": {
                "n_runs": len(seeds),
                "iid_comm_gain_mean": 0.1 * dim,
                "iid_comm_gain_std": 0.01 * dim,
                "ood_comm_gain_mean": 0.2 * dim,
                "ood_comm_gain_std": 0.02 * dim,
            },
        }

    monkeypatch.setattr("sweep_message_dim.run_seed_sweep", fake_run_seed_sweep)

    summaries = run_message_dim_sweep(
        base_config=CommTrainConfig(),
        message_dims=[4, 8],
        seeds=[42, 43],
    )

    assert calls == [(4, [42, 43]), (8, [42, 43])]
    assert summaries == [
        {
            "message_dim": 4,
            "n_seeds": 2,
            "ood_signature_seed": 42,
            "iid_comm_gain_mean": 0.4,
            "iid_comm_gain_std": 0.04,
            "ood_comm_gain_mean": 0.8,
            "ood_comm_gain_std": 0.08,
        },
        {
            "message_dim": 8,
            "n_seeds": 2,
            "ood_signature_seed": 42,
            "iid_comm_gain_mean": 0.8,
            "iid_comm_gain_std": 0.08,
            "ood_comm_gain_mean": 1.6,
            "ood_comm_gain_std": 0.16,
        },
    ]

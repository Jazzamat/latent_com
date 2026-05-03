import pytest

from sweep_comm import SWEEP_METRIC_KEYS, aggregate_float_metrics, run_seed_sweep
from train_comm import CommTrainConfig


def test_aggregate_float_metrics_computes_mean_std_min_max() -> None:
    run_results = [
        {
            "iid_comm_gain": 0.2,
            "ood_comm_gain": 0.5,
        },
        {
            "iid_comm_gain": 0.4,
            "ood_comm_gain": 0.9,
        },
    ]
    aggregate = aggregate_float_metrics(run_results, metric_keys=("iid_comm_gain", "ood_comm_gain"))

    assert aggregate["n_runs"] == 2
    assert aggregate["iid_comm_gain_mean"] == pytest.approx(0.3)
    assert aggregate["iid_comm_gain_std"] == pytest.approx(0.1)
    assert aggregate["iid_comm_gain_min"] == 0.2
    assert aggregate["iid_comm_gain_max"] == 0.4
    assert aggregate["ood_comm_gain_mean"] == pytest.approx(0.7)
    assert aggregate["ood_comm_gain_std"] == pytest.approx(0.2)
    assert aggregate["ood_comm_gain_min"] == 0.5
    assert aggregate["ood_comm_gain_max"] == 0.9


def test_run_seed_sweep_reports_aggregate_metrics() -> None:
    config = CommTrainConfig(
        device="cpu",
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=64,
        val_size=16,
        ood_val_size=16,
        ood_holdout_ratio=0.25,
        d_model=16,
        hidden_size=16,
        message_dim=8,
        batch_size=16,
        epochs=1,
        print_every=1,
    )

    sweep = run_seed_sweep(base_config=config, seeds=[3, 4])

    assert sweep["seeds"] == [3, 4]
    assert sweep["ood_signature_seed"] == 42
    assert len(sweep["runs"]) == 2
    assert sweep["aggregate"]["n_runs"] == 2

    heldout_sets = {tuple(run["heldout_visibility_signatures"]) for run in sweep["runs"]}
    assert len(heldout_sets) == 1

    for key in SWEEP_METRIC_KEYS:
        assert f"{key}_mean" in sweep["aggregate"]
        assert f"{key}_std" in sweep["aggregate"]
        assert f"{key}_min" in sweep["aggregate"]
        assert f"{key}_max" in sweep["aggregate"]

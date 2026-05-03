from train_comm import CommTrainConfig, run_experiment


def test_run_experiment_reports_iid_and_ood_metrics() -> None:
    config = CommTrainConfig(
        seed=7,
        device="cpu",
        n_locations=4,
        n_moves=3,
        protagonist_sees_prob=0.5,
        train_size=128,
        val_size=32,
        ood_val_size=32,
        ood_holdout_ratio=0.25,
        d_model=32,
        hidden_size=32,
        message_dim=16,
        batch_size=32,
        epochs=1,
        print_every=1,
    )

    results = run_experiment(config)

    expected_keys = {
        "device",
        "train_size",
        "val_size",
        "ood_val_size",
        "ood_holdout_ratio",
        "ood_signature_seed",
        "heldout_visibility_signatures",
        "iid_with_message_acc",
        "iid_zero_message_acc",
        "iid_shuffled_message_acc",
        "iid_comm_gain",
        "ood_with_message_acc",
        "ood_zero_message_acc",
        "ood_shuffled_message_acc",
        "ood_comm_gain",
    }
    assert expected_keys.issubset(results.keys())

    heldout = results["heldout_visibility_signatures"]
    assert isinstance(heldout, list)
    assert len(heldout) >= 1

    for key in [
        "iid_with_message_acc",
        "iid_zero_message_acc",
        "iid_shuffled_message_acc",
        "ood_with_message_acc",
        "ood_zero_message_acc",
        "ood_shuffled_message_acc",
    ]:
        value = results[key]
        assert 0.0 <= value <= 1.0

    assert -1.0 <= results["iid_comm_gain"] <= 1.0
    assert -1.0 <= results["ood_comm_gain"] <= 1.0

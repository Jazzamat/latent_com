import dataclasses

import config


def test_model_config_defaults() -> None:
    cfg = config.ModelConfig(vocab_size=23)
    assert dataclasses.is_dataclass(cfg)
    assert cfg.vocab_size == 23
    assert cfg.block_size == 128
    assert cfg.n_layers == 4
    assert cfg.n_heads == 4
    assert cfg.d_model == 256
    assert cfg.mlp_hidden == 1024
    assert cfg.dropout == 0.1


def test_train_config_has_fields_used_by_runtime() -> None:
    assert hasattr(config, "TrainConfig"), "config.TrainConfig is required by train.py and generate.py"

    train_cfg = config.TrainConfig()
    required_fields = [
        "seed",
        "device",
        "text_path",
        "train_split",
        "batch_size",
        "max_iters",
        "eval_interval",
        "eval_iters",
        "learning_rate",
        "weight_decay",
        "grad_clip",
        "checkpoint_path",
    ]

    for field_name in required_fields:
        assert hasattr(train_cfg, field_name), f"TrainConfig missing field: {field_name}"

    assert 0.0 < train_cfg.train_split < 1.0
    assert train_cfg.batch_size > 0
    assert train_cfg.max_iters > 0
    assert train_cfg.eval_interval > 0
    assert train_cfg.eval_iters > 0

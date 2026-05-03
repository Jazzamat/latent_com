import os
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GPU_OPT_IN_ENV = "TINYLM_BLACKBOX_USE_GPU"


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session")
def blackbox_device() -> str:
    use_gpu = _is_truthy(os.getenv(GPU_OPT_IN_ENV, ""))
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"

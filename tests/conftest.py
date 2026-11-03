"""Make the package and test helpers importable however pytest is invoked, and share fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pytest  # noqa: E402
import torch  # noqa: E402

from tiny import TinyTokenizer, TinyWorld  # noqa: E402


@pytest.fixture
def tokenizer() -> TinyTokenizer:
    return TinyTokenizer()


@pytest.fixture
def world() -> TinyWorld:
    return TinyWorld()


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(1234)

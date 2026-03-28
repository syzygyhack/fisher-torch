"""Shared test fixtures for fisher-torch."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy RNG."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_simplex_1d(rng: np.random.Generator) -> np.ndarray:
    """A valid 5-component simplex vector."""
    raw = rng.dirichlet(np.ones(5))
    return raw.astype(np.float64)


@pytest.fixture
def sample_simplex_batch(rng: np.random.Generator) -> np.ndarray:
    """A batch of 4 valid 5-component simplex vectors."""
    raw = rng.dirichlet(np.ones(5), size=4)
    return raw.astype(np.float64)


@pytest.fixture
def sample_tensor_1d() -> torch.Tensor:
    """A valid 5-component probability tensor."""
    return torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15])


@pytest.fixture
def sample_tensor_batch() -> torch.Tensor:
    """A batch of 3 valid probability tensors."""
    return torch.tensor(
        [[0.1, 0.2, 0.3, 0.25, 0.15],
         [0.4, 0.1, 0.1, 0.2, 0.2],
         [0.05, 0.05, 0.8, 0.05, 0.05]]
    )


@pytest.fixture
def sample_logits() -> torch.Tensor:
    """Logits for a 10-class vocabulary."""
    torch.manual_seed(42)
    return torch.randn(10)


@pytest.fixture
def sample_logits_batch() -> torch.Tensor:
    """Batch of logits for a 10-class vocabulary."""
    torch.manual_seed(42)
    return torch.randn(4, 10)

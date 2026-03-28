"""Tests for fisher_torch.convert."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fisher_torch.convert import from_simplex_array, to_simplex_array


class TestToSimplexArray:
    """Tests for to_simplex_array."""

    def test_produces_float64(self):
        t = torch.tensor([0.2, 0.3, 0.5])
        arr = to_simplex_array(t)
        assert arr.dtype == np.float64

    def test_warns_on_sum_drift(self):
        t = torch.tensor([0.5, 0.5, 0.5])  # sums to 1.5
        with pytest.warns(UserWarning, match="drift"):
            arr = to_simplex_array(t)
        # After renormalization, should sum to 1.
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-12)

    def test_rejects_negative(self):
        t = torch.tensor([-0.5, 0.5, 1.0])
        with pytest.raises(ValueError):
            to_simplex_array(t)

    def test_single_vector(self):
        t = torch.tensor([0.1, 0.2, 0.7])
        arr = to_simplex_array(t)
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr.sum(), 1.0, atol=1e-12)

    def test_batch(self):
        t = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        arr = to_simplex_array(t)
        assert arr.shape == (2, 3)
        np.testing.assert_allclose(arr.sum(axis=-1), 1.0, atol=1e-12)

    def test_detaches_grad(self):
        t = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
        arr = to_simplex_array(t)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64


class TestFromSimplexArray:
    """Tests for from_simplex_array."""

    def test_default_dtype_float32(self):
        arr = np.array([0.2, 0.3, 0.5])
        t = from_simplex_array(arr)
        assert t.dtype == torch.float32

    def test_rejects_strongly_negative(self):
        arr = np.array([-1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            from_simplex_array(arr)

    def test_single_vector(self):
        arr = np.array([0.1, 0.2, 0.7])
        t = from_simplex_array(arr)
        assert t.shape == (3,)

    def test_batch(self):
        arr = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
        t = from_simplex_array(arr)
        assert t.shape == (2, 3)

    def test_device_param(self):
        arr = np.array([0.2, 0.3, 0.5])
        t = from_simplex_array(arr, device="cpu")
        assert t.device == torch.device("cpu")

    def test_dtype_param(self):
        arr = np.array([0.2, 0.3, 0.5])
        t = from_simplex_array(arr, dtype=torch.float64)
        assert t.dtype == torch.float64


class TestRoundtrip:
    """Roundtrip conversion tests."""

    def test_tensor_to_numpy_to_tensor(self):
        original = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        arr = to_simplex_array(original)
        recovered = from_simplex_array(arr, dtype=torch.float64)
        torch.testing.assert_close(recovered, original)

    def test_roundtrip_batch(self):
        original = torch.tensor(
            [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
            dtype=torch.float64,
        )
        arr = to_simplex_array(original)
        recovered = from_simplex_array(arr, dtype=torch.float64)
        torch.testing.assert_close(recovered, original)

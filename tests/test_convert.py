"""Tests for fisher_torch.convert."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fisher_torch.convert import (
    from_simplex_array,
    stack_attention,
    to_simplex_array,
    truncate_and_renormalize,
)


class TestToSimplexArray:
    """Tests for to_simplex_array."""

    def test_produces_float64(self):
        t = torch.tensor([0.2, 0.3, 0.5])
        arr = to_simplex_array(t)
        assert arr.dtype == np.float64

    def test_silently_renormalizes_sum_drift(self):
        """to_simplex_array silently renormalizes mild sum drift."""
        t = torch.tensor([0.5, 0.5, 0.5])  # sums to 1.5
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


class TestStackAttention:
    """Tests for stack_attention."""

    def test_single_position_squeeze(self):
        """With 1 position per head, result is (n_layers, n_heads, seq_len)."""
        n_heads, seq_len = 4, 10
        attn = {
            0: np.random.dirichlet(np.ones(seq_len), size=n_heads),
            2: np.random.dirichlet(np.ones(seq_len), size=n_heads),
        }
        result = stack_attention(attn, n_heads=n_heads)
        assert result.shape == (2, n_heads, seq_len)

    def test_multi_position_no_squeeze(self):
        """With >1 position, result is (n_layers, n_heads, n_positions, seq_len)."""
        n_heads, n_positions, seq_len = 4, 3, 10
        attn = {
            0: np.random.dirichlet(
                np.ones(seq_len), size=n_heads * n_positions
            ),
        }
        result = stack_attention(attn, n_heads=n_heads)
        assert result.shape == (1, n_heads, n_positions, seq_len)

    def test_values_preserved(self):
        """Values survive the reshape correctly."""
        n_heads = 2
        layer0 = np.array([
            [0.5, 0.1, 0.1, 0.1, 0.2],  # head 0
            [0.2, 0.2, 0.2, 0.2, 0.2],  # head 1
        ])
        attn = {0: layer0}
        result = stack_attention(attn, n_heads=n_heads)
        np.testing.assert_array_equal(result[0, 0], layer0[0])
        np.testing.assert_array_equal(result[0, 1], layer0[1])

    def test_layer_order_sorted(self):
        """Layers appear in sorted key order regardless of dict insertion."""
        n_heads, seq_len = 2, 4
        layer5 = np.random.dirichlet(np.ones(seq_len), size=n_heads)
        layer2 = np.random.dirichlet(np.ones(seq_len), size=n_heads)
        attn = {5: layer5, 2: layer2}
        result = stack_attention(attn, n_heads=n_heads)
        # Index 0 should be layer 2, index 1 should be layer 5
        np.testing.assert_array_equal(result[0], layer2.reshape(n_heads, seq_len))
        np.testing.assert_array_equal(result[1], layer5.reshape(n_heads, seq_len))


class TestTruncateAndRenormalize:
    """Tests for truncate_and_renormalize."""

    def test_truncate_shorter(self):
        """Truncate from length 10 to 5."""
        rng = np.random.default_rng(42)
        arr = rng.dirichlet(np.ones(10), size=3)
        result = truncate_and_renormalize(arr, 5)
        assert result.shape == (3, 5)

    def test_renormalize_after_truncate(self):
        """Rows should sum to 1 after truncation."""
        rng = np.random.default_rng(42)
        arr = rng.dirichlet(np.ones(10), size=3)
        result = truncate_and_renormalize(arr, 5)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-12)

    def test_null_row_uniform_fallback(self):
        """A zero row should become uniform after truncation."""
        arr = np.zeros((2, 10))
        arr[0] = np.ones(10) / 10  # valid row
        # arr[1] stays all-zero
        result = truncate_and_renormalize(arr, 5)
        np.testing.assert_allclose(result[1], 1.0 / 5, atol=1e-12)

    def test_no_truncation_when_same_length(self):
        """If target_len == array length, just copy (already valid)."""
        rng = np.random.default_rng(42)
        arr = rng.dirichlet(np.ones(5), size=3)
        result = truncate_and_renormalize(arr, 5)
        np.testing.assert_allclose(result, arr, atol=1e-12)

    def test_1d_input(self):
        """Single simplex vector (1D) should work."""
        arr = np.array([0.4, 0.3, 0.2, 0.1])
        result = truncate_and_renormalize(arr, 2)
        assert result.shape == (2,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-12)

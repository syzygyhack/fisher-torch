"""Tests for fisher_torch.utils."""

from __future__ import annotations

import pytest
import torch

from fisher_torch.utils import safe_softmax, topk_to_simplex


class TestSafeSoftmax:
    """Tests for safe_softmax."""

    def test_sums_to_one(self):
        logits = torch.randn(10)
        probs = safe_softmax(logits)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_sums_to_one_batch(self):
        logits = torch.randn(4, 10)
        probs = safe_softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)

    def test_temperature_higher_more_uniform(self):
        logits = torch.tensor([10.0, 0.0, 0.0])
        low_temp = safe_softmax(logits, temperature=0.1)
        high_temp = safe_softmax(logits, temperature=10.0)
        # Higher temperature → more uniform → higher entropy
        # Max probability should be lower at higher temperature.
        assert high_temp.max().item() < low_temp.max().item()

    def test_numerically_stable_large_logits(self):
        logits = torch.tensor([1000.0, 999.0, 998.0])
        probs = safe_softmax(logits)
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_all_equal_logits(self):
        logits = torch.ones(5)
        probs = safe_softmax(logits)
        expected = torch.full((5,), 0.2)
        torch.testing.assert_close(probs, expected, atol=1e-6, rtol=1e-6)

    def test_dim_parameter(self):
        logits = torch.randn(3, 5)
        probs = safe_softmax(logits, dim=0)
        sums = probs.sum(dim=0)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)


class TestTopkToSimplex:
    """Tests for topk_to_simplex."""

    def _make_probs(self):
        """Return a simple probability vector."""
        return torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])

    def test_single_remainder_shape(self):
        probs = self._make_probs()
        result = topk_to_simplex(probs, top_k=3, mode="single_remainder")
        assert result.shape == (4,)  # K + 1

    def test_single_remainder_sums_to_one(self):
        probs = self._make_probs()
        result = topk_to_simplex(probs, top_k=3, mode="single_remainder")
        assert result.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_renormalize_shape(self):
        probs = self._make_probs()
        result = topk_to_simplex(probs, top_k=3, mode="renormalize")
        assert result.shape == (3,)  # K

    def test_renormalize_sums_to_one(self):
        probs = self._make_probs()
        result = topk_to_simplex(probs, top_k=3, mode="renormalize")
        assert result.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_known_tail_shape(self):
        probs = self._make_probs()
        result = topk_to_simplex(
            probs, top_k=3, mode="known_tail", tail_cardinality=5
        )
        assert result.shape == (8,)  # K + tail_cardinality

    def test_known_tail_sums_to_one(self):
        probs = self._make_probs()
        result = topk_to_simplex(
            probs, top_k=3, mode="known_tail", tail_cardinality=5
        )
        assert result.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_known_tail_requires_cardinality(self):
        probs = self._make_probs()
        with pytest.raises(ValueError, match="tail_cardinality"):
            topk_to_simplex(probs, top_k=3, mode="known_tail")

    def test_invalid_mode(self):
        probs = self._make_probs()
        with pytest.raises(ValueError, match="Invalid mode"):
            topk_to_simplex(probs, top_k=3, mode="bogus")

    def test_batch_single_remainder(self):
        probs = torch.tensor(
            [[0.4, 0.3, 0.15, 0.1, 0.05], [0.5, 0.2, 0.15, 0.1, 0.05]]
        )
        result = topk_to_simplex(probs, top_k=2, mode="single_remainder")
        assert result.shape == (2, 3)  # (batch, K+1)
        sums = result.sum(dim=-1)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_renormalize(self):
        probs = torch.tensor(
            [[0.4, 0.3, 0.15, 0.1, 0.05], [0.5, 0.2, 0.15, 0.1, 0.05]]
        )
        result = topk_to_simplex(probs, top_k=2, mode="renormalize")
        assert result.shape == (2, 2)
        sums = result.sum(dim=-1)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_known_tail(self):
        probs = torch.tensor(
            [[0.4, 0.3, 0.15, 0.1, 0.05], [0.5, 0.2, 0.15, 0.1, 0.05]]
        )
        result = topk_to_simplex(
            probs, top_k=2, mode="known_tail", tail_cardinality=3
        )
        assert result.shape == (2, 5)  # (batch, K + tail_cardinality)
        sums = result.sum(dim=-1)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)

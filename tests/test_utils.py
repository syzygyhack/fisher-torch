"""Tests for fisher_torch.utils."""

from __future__ import annotations

import pytest
import torch

from fisher_torch.utils import ProjectionSpec, safe_softmax, topk_softmax


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
        # Higher temperature -> more uniform -> lower max probability.
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

    def test_no_negative_values(self):
        logits = torch.randn(100)
        probs = safe_softmax(logits)
        assert (probs >= 0).all()

    def test_clips_near_zero(self):
        # Very peaked distribution: most entries should be clipped to 0.
        logits = torch.zeros(1000)
        logits[0] = 100.0
        probs = safe_softmax(logits)
        # The non-top entry should be exactly 0, not a tiny float.
        assert (probs[1:] == 0.0).all()


class TestTopkSoftmax:
    """Tests for topk_softmax."""

    def test_single_remainder_shape(self):
        logits = torch.randn(10)
        simplex, spec = topk_softmax(logits, 3, remainder_mode="single_remainder")
        assert simplex.shape == (4,)  # K + 1
        assert spec.k == 3
        assert spec.remainder_mode == "single_remainder"

    def test_single_remainder_sums_to_one(self):
        logits = torch.randn(10)
        simplex, _ = topk_softmax(logits, 3, remainder_mode="single_remainder")
        assert simplex.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_renormalize_shape(self):
        logits = torch.randn(10)
        simplex, spec = topk_softmax(logits, 3, remainder_mode="renormalize")
        assert simplex.shape == (3,)  # K
        assert spec.remainder_mode == "renormalize"

    def test_renormalize_sums_to_one(self):
        logits = torch.randn(10)
        simplex, _ = topk_softmax(logits, 3, remainder_mode="renormalize")
        assert simplex.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_known_tail_shape(self):
        logits = torch.randn(10)
        simplex, spec = topk_softmax(
            logits, 3, remainder_mode="known_tail", tail_cardinality=5
        )
        assert simplex.shape == (8,)  # K + tail_cardinality
        assert spec.tail_cardinality == 5

    def test_known_tail_sums_to_one(self):
        logits = torch.randn(10)
        simplex, _ = topk_softmax(
            logits, 3, remainder_mode="known_tail", tail_cardinality=5
        )
        assert simplex.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_known_tail_requires_cardinality(self):
        logits = torch.randn(10)
        with pytest.raises(ValueError, match="tail_cardinality"):
            topk_softmax(logits, 3, remainder_mode="known_tail")

    def test_invalid_mode(self):
        logits = torch.randn(10)
        with pytest.raises(ValueError):
            topk_softmax(logits, 3, remainder_mode="bogus")

    def test_batch_single_remainder(self):
        logits = torch.randn(4, 10)
        simplex, spec = topk_softmax(logits, 2, remainder_mode="single_remainder")
        assert simplex.shape == (4, 3)  # (batch, K+1)
        sums = simplex.sum(dim=-1)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_renormalize(self):
        logits = torch.randn(4, 10)
        simplex, _ = topk_softmax(logits, 2, remainder_mode="renormalize")
        assert simplex.shape == (4, 2)
        sums = simplex.sum(dim=-1)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_known_tail(self):
        logits = torch.randn(4, 10)
        simplex, _ = topk_softmax(
            logits, 2, remainder_mode="known_tail", tail_cardinality=3
        )
        assert simplex.shape == (4, 5)  # (batch, K + tail_cardinality)
        sums = simplex.sum(dim=-1)
        for s in sums:
            assert s.item() == pytest.approx(1.0, abs=1e-6)

    def test_temperature_affects_result(self):
        logits = torch.tensor([10.0, 5.0, 1.0, 0.0, 0.0])
        s_low, _ = topk_softmax(logits, 3, temperature=0.5)
        s_high, _ = topk_softmax(logits, 3, temperature=5.0)
        # Higher temp -> more uniform top-k -> larger remainder (or more even)
        # The top entry should be smaller with higher temperature.
        assert s_high[0].item() < s_low[0].item()

    def test_returns_projection_spec(self):
        logits = torch.randn(20)
        _, spec = topk_softmax(logits, 5, remainder_mode="single_remainder")
        assert isinstance(spec, ProjectionSpec)
        assert spec.k == 5
        assert spec.remainder_mode == "single_remainder"
        assert spec.original_vocab_size == 20
        assert spec.tail_cardinality is None

    def test_non_negative(self):
        logits = torch.randn(50)
        simplex, _ = topk_softmax(logits, 10)
        assert (simplex >= 0).all()

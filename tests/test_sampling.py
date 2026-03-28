"""Tests for fisher_torch.sampling."""

from __future__ import annotations

import pytest

from fisher_torch.sampling import SamplingPolicy


class TestSamplingPolicyDefaults:
    """Test default policy behaviour."""

    def test_default_final_token_only(self):
        policy = SamplingPolicy()
        assert policy.final_token_only is True

    def test_default_remainder_mode(self):
        policy = SamplingPolicy()
        assert policy.remainder_mode == "single_remainder"


class TestSelectedLayers:
    """Tests for selected_layers."""

    def test_explicit_list(self):
        policy = SamplingPolicy(layers=[0, 2, 4])
        assert policy.selected_layers(6) == [0, 2, 4]

    def test_stride(self):
        policy = SamplingPolicy(layer_stride=3)
        assert policy.selected_layers(10) == [0, 3, 6, 9]

    def test_default_all(self):
        policy = SamplingPolicy()
        assert policy.selected_layers(4) == [0, 1, 2, 3]

    def test_explicit_list_takes_priority_over_stride(self):
        policy = SamplingPolicy(layers=[1, 3], layer_stride=2)
        assert policy.selected_layers(6) == [1, 3]

    def test_out_of_bounds(self):
        policy = SamplingPolicy(layers=[0, 10])
        with pytest.raises(ValueError, match="out of bounds"):
            policy.selected_layers(5)

    def test_negative_index_out_of_bounds(self):
        policy = SamplingPolicy(layers=[-1])
        with pytest.raises(ValueError, match="out of bounds"):
            policy.selected_layers(5)


class TestSelectedHeads:
    """Tests for selected_heads."""

    def test_explicit_list(self):
        policy = SamplingPolicy(heads=[0, 3, 7])
        assert policy.selected_heads(8) == [0, 3, 7]

    def test_stride(self):
        policy = SamplingPolicy(head_stride=4)
        assert policy.selected_heads(12) == [0, 4, 8]

    def test_default_all(self):
        policy = SamplingPolicy()
        assert policy.selected_heads(3) == [0, 1, 2]

    def test_out_of_bounds(self):
        policy = SamplingPolicy(heads=[5])
        with pytest.raises(ValueError, match="out of bounds"):
            policy.selected_heads(4)


class TestSelectedPositions:
    """Tests for selected_positions."""

    def test_final_token_only(self):
        policy = SamplingPolicy(final_token_only=True)
        assert policy.selected_positions(10) == [9]

    def test_explicit_list(self):
        policy = SamplingPolicy(final_token_only=False, positions=[0, 5, 9])
        assert policy.selected_positions(10) == [0, 5, 9]

    def test_slice(self):
        policy = SamplingPolicy(final_token_only=False, positions=slice(2, 6))
        assert policy.selected_positions(10) == [2, 3, 4, 5]

    def test_slice_with_step(self):
        policy = SamplingPolicy(
            final_token_only=False, positions=slice(0, 10, 3)
        )
        assert policy.selected_positions(10) == [0, 3, 6, 9]

    def test_default_all(self):
        policy = SamplingPolicy(final_token_only=False)
        assert policy.selected_positions(4) == [0, 1, 2, 3]

    def test_max_tokens_per_sample(self):
        policy = SamplingPolicy(
            final_token_only=False, max_tokens_per_sample=3
        )
        result = policy.selected_positions(10)
        assert len(result) == 3
        assert result == [0, 1, 2]

    def test_max_tokens_with_slice(self):
        policy = SamplingPolicy(
            final_token_only=False,
            positions=slice(0, 10),
            max_tokens_per_sample=5,
        )
        result = policy.selected_positions(10)
        assert len(result) == 5
        assert result == [0, 1, 2, 3, 4]

    def test_out_of_bounds(self):
        policy = SamplingPolicy(final_token_only=False, positions=[0, 20])
        with pytest.raises(ValueError, match="out of bounds"):
            policy.selected_positions(10)

    def test_final_token_only_overrides_positions(self):
        policy = SamplingPolicy(
            final_token_only=True, positions=[0, 1, 2]
        )
        assert policy.selected_positions(10) == [9]

    def test_empty_sequence_returns_empty(self):
        """seq_len=0 should return empty list, not [-1]."""
        policy = SamplingPolicy(final_token_only=True)
        assert policy.selected_positions(0) == []

    def test_empty_sequence_all_positions(self):
        policy = SamplingPolicy(final_token_only=False)
        assert policy.selected_positions(0) == []

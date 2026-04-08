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

    def test_default_tail_cardinality(self):
        policy = SamplingPolicy()
        assert policy.tail_cardinality is None

    def test_tail_cardinality_set(self):
        policy = SamplingPolicy(
            top_k=10,
            remainder_mode="known_tail",
            tail_cardinality=990,
        )
        assert policy.tail_cardinality == 990


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


class TestPositionPresets:
    """Tests for position_preset feature."""

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown position_preset"):
            SamplingPolicy(position_preset="bogus")

    def test_atlas_positions_long_sequence(self):
        """Atlas preset on seq_len=100: early=4, mid=50, late=97, final=99."""
        policy = SamplingPolicy(position_preset="atlas")
        result = policy.selected_positions(100)
        assert result == [4, 50, 97, 99]

    def test_atlas_labels_long_sequence(self):
        policy = SamplingPolicy(position_preset="atlas")
        labels = policy.position_labels(100)
        assert labels == {"early": 4, "mid": 50, "late": 97, "final": 99}

    def test_atlas_overrides_final_token_only(self):
        """Preset should override the default final_token_only=True."""
        policy = SamplingPolicy(position_preset="atlas")
        assert policy.final_token_only is True  # default unchanged
        result = policy.selected_positions(100)
        assert len(result) == 4  # preset wins, not final-token-only

    def test_atlas_short_sequence_deduplicates(self):
        """Short sequence may collapse positions; duplicates are removed."""
        policy = SamplingPolicy(position_preset="atlas")
        result = policy.selected_positions(2)
        # All positions should be valid and unique
        assert len(result) == len(set(result))
        assert all(0 <= p < 2 for p in result)

    def test_atlas_single_token(self):
        policy = SamplingPolicy(position_preset="atlas")
        result = policy.selected_positions(1)
        assert result == [0]

    def test_quartiles_positions(self):
        policy = SamplingPolicy(position_preset="quartiles")
        result = policy.selected_positions(100)
        assert result == [25, 50, 75, 99]

    def test_quartiles_labels(self):
        policy = SamplingPolicy(position_preset="quartiles")
        labels = policy.position_labels(100)
        assert labels == {"q25": 25, "q50": 50, "q75": 75, "q100": 99}

    def test_position_labels_none_without_preset(self):
        policy = SamplingPolicy()
        assert policy.position_labels(100) is None

    def test_position_labels_none_empty_sequence(self):
        policy = SamplingPolicy(position_preset="atlas")
        assert policy.position_labels(0) is None

    def test_preset_with_max_tokens(self):
        policy = SamplingPolicy(
            position_preset="atlas", max_tokens_per_sample=2
        )
        result = policy.selected_positions(100)
        assert len(result) == 2

    def test_labels_respect_max_tokens(self):
        """position_labels must match selected_positions when capped."""
        policy = SamplingPolicy(
            position_preset="atlas", max_tokens_per_sample=2
        )
        positions = policy.selected_positions(100)
        labels = policy.position_labels(100)
        assert len(labels) == 2
        assert list(labels.values()) == positions

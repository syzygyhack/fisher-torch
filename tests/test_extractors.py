"""Tests for fisher_torch.extractors."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fisher_torch.extractors import (
    extract_attention,
    extract_hidden_states,
    extract_layerwise_predictions,
    extract_predictions,
    extract_routing,
)
from fisher_torch.sampling import SamplingPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
SEQ_LEN = 10
N_LAYERS = 4
N_HEADS = 4


def _make_logits(batch: int = 1) -> torch.Tensor:
    """Create random logits tensor (batch, seq_len, vocab_size)."""
    torch.manual_seed(0)
    logits = torch.randn(batch, SEQ_LEN, VOCAB_SIZE)
    if batch == 1:
        logits = logits.squeeze(0)  # (seq_len, vocab_size)
    return logits


def _make_attention(batch: int = 1) -> tuple[torch.Tensor, ...]:
    """Create attention weight tuple, one per layer.

    Each tensor: (batch, n_heads, seq_len, seq_len).
    Rows already sum to 1 (softmax output).
    """
    torch.manual_seed(1)
    layers = []
    for _ in range(N_LAYERS):
        raw = torch.randn(batch, N_HEADS, SEQ_LEN, SEQ_LEN)
        layers.append(torch.softmax(raw, dim=-1))
    return tuple(layers)


def _make_causal_attention() -> tuple[torch.Tensor, ...]:
    """Create attention with proper causal mask (zeros above diagonal).

    Each tensor: (1, n_heads, seq_len, seq_len).
    Position p only attends to positions 0..p.
    """
    torch.manual_seed(1)
    mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
    layers = []
    for _ in range(N_LAYERS):
        raw = torch.randn(1, N_HEADS, SEQ_LEN, SEQ_LEN)
        raw = raw.masked_fill(mask, float("-inf"))
        attn = torch.softmax(raw, dim=-1)
        layers.append(attn)
    return tuple(layers)


def _make_hidden_states(batch: int = 1) -> tuple[torch.Tensor, ...]:
    """Create hidden state tuple, one per layer.

    Each tensor: (batch, seq_len, hidden_dim).
    """
    torch.manual_seed(2)
    hidden_dim = 64
    return tuple(
        torch.randn(batch, SEQ_LEN, hidden_dim) for _ in range(N_LAYERS)
    )


def _make_lm_head() -> torch.nn.Linear:
    """Create a small lm_head: hidden_dim -> vocab_size."""
    torch.manual_seed(3)
    return torch.nn.Linear(64, VOCAB_SIZE, bias=False)


def _make_router_logits() -> torch.Tensor:
    """Create MoE router logits: (seq_len, n_experts)."""
    torch.manual_seed(4)
    n_experts = 8
    return torch.randn(SEQ_LEN, n_experts)


def _is_valid_simplex(arr: np.ndarray) -> None:
    """Assert arr is a valid float64 simplex array."""
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert (arr >= 0).all(), f"Negative values found: {arr.min()}"
    sums = arr.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# extract_predictions
# ---------------------------------------------------------------------------


class TestExtractPredictions:
    """Tests for extract_predictions."""

    def test_valid_simplex(self):
        logits = _make_logits()
        result = extract_predictions(logits)
        assert result.shape == (SEQ_LEN, VOCAB_SIZE)
        _is_valid_simplex(result)

    def test_batch_squeeze(self):
        """Batch=1 input should be squeezed to 2D."""
        logits = torch.randn(1, SEQ_LEN, VOCAB_SIZE)
        result = extract_predictions(logits)
        assert result.ndim == 2
        assert result.shape == (SEQ_LEN, VOCAB_SIZE)
        _is_valid_simplex(result)

    def test_top_k_shape(self):
        logits = _make_logits()
        result = extract_predictions(logits, top_k=10)
        assert result.shape == (SEQ_LEN, 11)  # K + 1 for single_remainder
        _is_valid_simplex(result)

    def test_temperature_more_uniform(self):
        logits = _make_logits()
        low_temp = extract_predictions(logits, temperature=0.1)
        high_temp = extract_predictions(logits, temperature=10.0)
        assert high_temp.max() < low_temp.max()

    def test_remainder_mode_renormalize(self):
        logits = _make_logits()
        result = extract_predictions(
            logits, top_k=10, remainder_mode="renormalize"
        )
        assert result.shape == (SEQ_LEN, 10)
        _is_valid_simplex(result)

    def test_multi_batch(self):
        """Batch > 1 should preserve batch dim."""
        logits = _make_logits(batch=3)
        result = extract_predictions(logits)
        assert result.shape == (3, SEQ_LEN, VOCAB_SIZE)
        for b in range(3):
            _is_valid_simplex(result[b])

    def test_known_tail_shape(self):
        logits = _make_logits()
        result = extract_predictions(
            logits, top_k=10, remainder_mode="known_tail",
            tail_cardinality=90,
        )
        assert result.shape == (SEQ_LEN, 100)
        _is_valid_simplex(result)

    def test_return_tensors(self):
        logits = _make_logits()
        result = extract_predictions(logits, return_tensors=True)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (SEQ_LEN, VOCAB_SIZE)

    def test_return_tensors_with_topk_raises(self):
        logits = _make_logits()
        with pytest.raises(ValueError, match="top_k is not supported"):
            extract_predictions(logits, top_k=10, return_tensors=True)


# ---------------------------------------------------------------------------
# extract_attention
# ---------------------------------------------------------------------------


class TestExtractAttention:
    """Tests for extract_attention."""

    def test_returns_dict(self):
        attn = _make_attention()
        result = extract_attention(attn)
        assert isinstance(result, dict)

    def test_keys_are_layer_indices(self):
        attn = _make_attention()
        result = extract_attention(attn)
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_values_are_valid_simplex(self):
        attn = _make_attention()
        result = extract_attention(attn)
        for arr in result.values():
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float64
            sums = arr.sum(axis=-1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_policy_selects_layers(self):
        attn = _make_attention()
        policy = SamplingPolicy(layers=[0, 2])
        result = extract_attention(attn, policy=policy)
        assert set(result.keys()) == {0, 2}

    def test_default_policy_final_token(self):
        attn = _make_attention()
        result = extract_attention(attn)
        for arr in result.values():
            assert arr.shape[-1] == SEQ_LEN

    def test_policy_selects_heads(self):
        attn = _make_attention()
        policy = SamplingPolicy(heads=[0, 2])
        result = extract_attention(attn, policy=policy)
        for arr in result.values():
            assert arr.shape[0] == 2

    def test_batch_gt1_returns_list(self):
        """Batch > 1 returns a list of dicts."""
        attn = _make_attention(batch=2)
        result = extract_attention(attn)
        assert isinstance(result, list)
        assert len(result) == 2
        for d in result:
            assert isinstance(d, dict)
            assert set(d.keys()) == {0, 1, 2, 3}

    def test_batch_gt1_valid_simplices(self):
        attn = _make_attention(batch=3)
        result = extract_attention(attn)
        for d in result:
            for arr in d.values():
                _is_valid_simplex(arr)

    def test_causal_trim_final_token_unchanged(self):
        attn = _make_causal_attention()
        result = extract_attention(attn, causal=True)
        for arr in result.values():
            assert arr.shape[-1] == SEQ_LEN
            _is_valid_simplex(arr)

    def test_causal_trim_non_final_position(self):
        attn = _make_causal_attention()
        policy = SamplingPolicy(final_token_only=False, positions=[4])
        result = extract_attention(attn, policy=policy, causal=True)
        for arr in result.values():
            assert arr.shape == (N_HEADS, 5)
            _is_valid_simplex(arr)

    def test_causal_false_preserves_full_row(self):
        attn = _make_causal_attention()
        policy = SamplingPolicy(final_token_only=False, positions=[4])
        result = extract_attention(attn, policy=policy, causal=False)
        for arr in result.values():
            assert arr.shape == (N_HEADS, SEQ_LEN)

    def test_return_tensors(self):
        attn = _make_attention()
        result = extract_attention(attn, return_tensors=True)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, torch.Tensor)

    def test_empty_heads_selection(self):
        """Empty heads list should return empty arrays, not crash."""
        attn = _make_attention()
        policy = SamplingPolicy(heads=[])
        result = extract_attention(attn, policy=policy)
        for arr in result.values():
            assert arr.shape[0] == 0

    def test_empty_positions_selection(self):
        """Empty positions should return empty arrays."""
        attn = _make_attention()
        policy = SamplingPolicy(
            final_token_only=False, positions=[],
        )
        result = extract_attention(attn, policy=policy)
        for arr in result.values():
            assert arr.shape[0] == 0


# ---------------------------------------------------------------------------
# extract_layerwise_predictions
# ---------------------------------------------------------------------------


class TestExtractLayerwisePredictions:
    """Tests for extract_layerwise_predictions."""

    def test_returns_list_of_dicts(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        result = extract_layerwise_predictions(hidden, lm_head)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_dict_has_required_keys(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        result = extract_layerwise_predictions(hidden, lm_head)
        for d in result:
            assert "layer" in d
            assert "predictions" in d

    def test_predictions_are_valid_simplex(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        result = extract_layerwise_predictions(hidden, lm_head)
        for d in result:
            _is_valid_simplex(d["predictions"])

    def test_policy_selects_layers(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        policy = SamplingPolicy(layers=[1, 3])
        result = extract_layerwise_predictions(hidden, lm_head, policy=policy)
        assert len(result) == 2
        assert result[0]["layer"] == 1
        assert result[1]["layer"] == 3

    def test_default_all_layers(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        result = extract_layerwise_predictions(hidden, lm_head)
        assert len(result) == N_LAYERS

    def test_top_k_via_policy(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        policy = SamplingPolicy(top_k=10)
        result = extract_layerwise_predictions(hidden, lm_head, policy=policy)
        for d in result:
            assert d["predictions"].shape[-1] == 11

    def test_batch_gt1_returns_list_of_lists(self):
        """Batch > 1 returns list[list[dict]]."""
        hidden = _make_hidden_states(batch=2)
        lm_head = _make_lm_head()
        result = extract_layerwise_predictions(hidden, lm_head)
        assert isinstance(result, list)
        assert len(result) == 2
        for batch_item in result:
            assert isinstance(batch_item, list)
            assert len(batch_item) == N_LAYERS

    def test_position_filtering_final_token(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        policy = SamplingPolicy(final_token_only=True)
        result = extract_layerwise_predictions(hidden, lm_head, policy=policy)
        for d in result:
            assert d["predictions"].shape[0] == 1
            _is_valid_simplex(d["predictions"])

    def test_position_filtering_explicit(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        policy = SamplingPolicy(
            final_token_only=False, positions=[1, 3, 5]
        )
        result = extract_layerwise_predictions(hidden, lm_head, policy=policy)
        for d in result:
            assert d["predictions"].shape[0] == 3
            _is_valid_simplex(d["predictions"])

    def test_return_tensors(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        result = extract_layerwise_predictions(
            hidden, lm_head, return_tensors=True, no_grad=False,
        )
        for d in result:
            assert isinstance(d["predictions"], torch.Tensor)

    def test_return_tensors_with_topk_raises(self):
        hidden = _make_hidden_states()
        lm_head = _make_lm_head()
        policy = SamplingPolicy(top_k=10)
        with pytest.raises(ValueError, match="top_k is not supported"):
            extract_layerwise_predictions(
                hidden, lm_head, policy=policy, return_tensors=True,
            )


# ---------------------------------------------------------------------------
# extract_hidden_states
# ---------------------------------------------------------------------------


class TestExtractHiddenStates:
    """Tests for extract_hidden_states."""

    def test_returns_list_of_dicts(self):
        hidden = _make_hidden_states()
        result = extract_hidden_states(hidden)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_dict_keys(self):
        hidden = _make_hidden_states()
        result = extract_hidden_states(hidden)
        for d in result:
            assert "layer" in d
            assert "hidden_states" in d

    def test_shape_final_token(self):
        """Default policy: final token only -> (1, hidden_dim) per layer."""
        hidden = _make_hidden_states()
        result = extract_hidden_states(hidden)
        for d in result:
            assert d["hidden_states"].shape == (1, 64)

    def test_shape_all_positions(self):
        hidden = _make_hidden_states()
        policy = SamplingPolicy(final_token_only=False)
        result = extract_hidden_states(hidden, policy=policy)
        for d in result:
            assert d["hidden_states"].shape == (SEQ_LEN, 64)

    def test_layer_selection(self):
        hidden = _make_hidden_states()
        policy = SamplingPolicy(layers=[0, 2])
        result = extract_hidden_states(hidden, policy=policy)
        assert len(result) == 2
        assert result[0]["layer"] == 0
        assert result[1]["layer"] == 2

    def test_position_selection(self):
        hidden = _make_hidden_states()
        policy = SamplingPolicy(final_token_only=False, positions=[1, 5])
        result = extract_hidden_states(hidden, policy=policy)
        for d in result:
            assert d["hidden_states"].shape == (2, 64)

    def test_numpy_dtype(self):
        hidden = _make_hidden_states()
        result = extract_hidden_states(hidden)
        for d in result:
            assert isinstance(d["hidden_states"], np.ndarray)
            assert d["hidden_states"].dtype == np.float64

    def test_return_tensors(self):
        hidden = _make_hidden_states()
        result = extract_hidden_states(hidden, return_tensors=True)
        for d in result:
            assert isinstance(d["hidden_states"], torch.Tensor)

    def test_batch_gt1_returns_list_of_lists(self):
        hidden = _make_hidden_states(batch=2)
        result = extract_hidden_states(hidden)
        assert isinstance(result, list)
        assert len(result) == 2
        for batch_item in result:
            assert isinstance(batch_item, list)
            assert len(batch_item) == N_LAYERS


# ---------------------------------------------------------------------------
# extract_routing
# ---------------------------------------------------------------------------


class TestExtractRouting:
    """Tests for extract_routing."""

    def test_valid_simplex(self):
        logits = _make_router_logits()
        result = extract_routing(logits)
        _is_valid_simplex(result)

    def test_shape(self):
        logits = _make_router_logits()
        result = extract_routing(logits)
        assert result.shape == (SEQ_LEN, 8)

    def test_tuple_input(self):
        """Tuple of router logits should be concatenated."""
        t1 = torch.randn(5, 8)
        t2 = torch.randn(3, 8)
        result = extract_routing((t1, t2))
        assert result.shape == (8, 8)
        _is_valid_simplex(result)

    def test_return_tensors(self):
        logits = _make_router_logits()
        result = extract_routing(logits, return_tensors=True)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (SEQ_LEN, 8)

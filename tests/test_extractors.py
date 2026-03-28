"""Tests for fisher_torch.extractors."""

from __future__ import annotations

import numpy as np
import torch

from fisher_torch.extractors import (
    extract_attention,
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


def _make_attention() -> tuple[torch.Tensor, ...]:
    """Create attention weight tuple, one per layer.

    Each tensor: (1, n_heads, seq_len, seq_len).
    Rows already sum to 1 (softmax output).
    """
    torch.manual_seed(1)
    layers = []
    for _ in range(N_LAYERS):
        raw = torch.randn(1, N_HEADS, SEQ_LEN, SEQ_LEN)
        attn = torch.softmax(raw, dim=-1)
        layers.append(attn)
    return tuple(layers)


def _make_hidden_states() -> tuple[torch.Tensor, ...]:
    """Create hidden state tuple, one per layer.

    Each tensor: (1, seq_len, hidden_dim).
    """
    torch.manual_seed(2)
    hidden_dim = 64
    return tuple(
        torch.randn(1, SEQ_LEN, hidden_dim) for _ in range(N_LAYERS)
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
        # Higher temperature → more uniform → lower max probability
        assert high_temp.max() < low_temp.max()

    def test_remainder_mode_renormalize(self):
        logits = _make_logits()
        result = extract_predictions(
            logits, top_k=10, remainder_mode="renormalize"
        )
        assert result.shape == (SEQ_LEN, 10)  # K only
        _is_valid_simplex(result)

    def test_multi_batch(self):
        """Batch > 1 should preserve batch dim."""
        logits = _make_logits(batch=3)
        result = extract_predictions(logits)
        assert result.shape == (3, SEQ_LEN, VOCAB_SIZE)
        # Each row should be a valid simplex
        for b in range(3):
            _is_valid_simplex(result[b])


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
        # Default policy: all layers
        assert set(result.keys()) == {0, 1, 2, 3}

    def test_values_are_valid_simplex(self):
        attn = _make_attention()
        result = extract_attention(attn)
        for layer_idx, arr in result.items():
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float64
            # Each row should sum to 1
            sums = arr.sum(axis=-1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_policy_selects_layers(self):
        attn = _make_attention()
        policy = SamplingPolicy(layers=[0, 2])
        result = extract_attention(attn, policy=policy)
        assert set(result.keys()) == {0, 2}

    def test_default_policy_final_token(self):
        """Default policy: final_token_only → one position per layer."""
        attn = _make_attention()
        result = extract_attention(attn)
        for arr in result.values():
            # With all heads and final_token_only, we get (n_heads, seq_len)
            assert arr.shape[-1] == SEQ_LEN

    def test_policy_selects_heads(self):
        attn = _make_attention()
        policy = SamplingPolicy(heads=[0, 2])
        result = extract_attention(attn, policy=policy)
        for arr in result.values():
            # Only 2 heads selected, with final_token_only → (2, seq_len)
            assert arr.shape[0] == 2


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
            assert d["predictions"].shape[-1] == 11  # K + 1


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
        assert result.shape == (8, 8)  # 5 + 3 tokens
        _is_valid_simplex(result)

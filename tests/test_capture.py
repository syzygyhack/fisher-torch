"""Tests for fisher_torch.capture."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch

from fisher_torch.capture import CaptureResult, ProjectionSpec, capture_forward
from fisher_torch.sampling import SamplingPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
SEQ_LEN = 10
N_LAYERS = 4
N_HEADS = 4
HIDDEN_DIM = 64


def _mock_model_output(
    *,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
):
    """Build a mock model output object."""
    torch.manual_seed(42)
    output = MagicMock()
    output.logits = torch.randn(1, SEQ_LEN, VOCAB_SIZE)

    if output_attentions:
        layers = []
        for _ in range(N_LAYERS):
            raw = torch.randn(1, N_HEADS, SEQ_LEN, SEQ_LEN)
            layers.append(torch.softmax(raw, dim=-1))
        output.attentions = tuple(layers)
    else:
        output.attentions = None

    if output_hidden_states:
        output.hidden_states = tuple(
            torch.randn(1, SEQ_LEN, HIDDEN_DIM) for _ in range(N_LAYERS + 1)
        )
    else:
        output.hidden_states = None

    return output


def _mock_model(
    *,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
):
    """Build a mock PreTrainedModel."""
    model = MagicMock()

    # Config
    model.config = MagicMock()
    model.config.vocab_size = VOCAB_SIZE
    model.config.num_hidden_layers = N_LAYERS
    model.config.num_attention_heads = N_HEADS
    model.config.output_attentions = False
    model.config.output_hidden_states = False

    # lm_head
    torch.manual_seed(42)
    model.lm_head = torch.nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    # Input embeddings (for get_input_device resolution)
    model.get_input_embeddings = MagicMock(
        return_value=torch.nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
    )

    # device
    type(model).device = PropertyMock(return_value=torch.device("cpu"))

    # Forward pass
    def forward_fn(input_ids, **kwargs):
        return _mock_model_output(
            output_attentions=kwargs.get("output_attentions", False)
            or model.config.output_attentions,
            output_hidden_states=kwargs.get("output_hidden_states", False)
            or model.config.output_hidden_states,
        )

    model.side_effect = forward_fn
    model.__call__ = forward_fn
    return model


# ---------------------------------------------------------------------------
# ProjectionSpec
# ---------------------------------------------------------------------------


class TestProjectionSpec:
    """Tests for ProjectionSpec."""

    def test_compatible_same(self):
        a = ProjectionSpec(mode="full", simplex_dim=100)
        b = ProjectionSpec(mode="full", simplex_dim=100)
        assert a.is_compatible_with(b)

    def test_incompatible_mode(self):
        a = ProjectionSpec(mode="full", simplex_dim=100)
        b = ProjectionSpec(mode="single_remainder", simplex_dim=100)
        assert not a.is_compatible_with(b)

    def test_incompatible_dim(self):
        a = ProjectionSpec(mode="full", simplex_dim=100)
        b = ProjectionSpec(mode="full", simplex_dim=50)
        assert not a.is_compatible_with(b)

    def test_compatible_with_top_k(self):
        a = ProjectionSpec(mode="single_remainder", top_k=10, simplex_dim=11)
        b = ProjectionSpec(mode="single_remainder", top_k=10, simplex_dim=11)
        assert a.is_compatible_with(b)

    def test_incompatible_top_k(self):
        a = ProjectionSpec(mode="single_remainder", top_k=10, simplex_dim=11)
        b = ProjectionSpec(mode="single_remainder", top_k=20, simplex_dim=21)
        assert not a.is_compatible_with(b)

    def test_frozen(self):
        spec = ProjectionSpec(mode="full", simplex_dim=100)
        with pytest.raises(AttributeError):
            spec.mode = "other"


# ---------------------------------------------------------------------------
# CaptureResult
# ---------------------------------------------------------------------------


class TestCaptureResult:
    """Tests for CaptureResult dataclass."""

    def test_defaults_none(self):
        result = CaptureResult()
        assert result.predictions is None
        assert result.attention is None
        assert result.hidden_states is None
        assert result.routing is None
        assert result.projection_spec is None
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# capture_forward
# ---------------------------------------------------------------------------


class TestCaptureForward:
    """Tests for capture_forward."""

    def test_predictions_not_none(self):
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(model, input_ids, predictions=True)
        assert result.predictions is not None
        assert isinstance(result.predictions, np.ndarray)
        assert result.predictions.dtype == np.float64

    def test_attention_not_none(self):
        model = _mock_model(output_attentions=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(
            model, input_ids, predictions=False, attention=True
        )
        assert result.attention is not None
        assert isinstance(result.attention, dict)

    def test_hidden_states_not_none(self):
        model = _mock_model(output_hidden_states=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(
            model, input_ids, predictions=False, hidden_states=True
        )
        assert result.hidden_states is not None
        assert isinstance(result.hidden_states, list)

    def test_restores_config(self):
        model = _mock_model()
        model.config.output_attentions = False
        model.config.output_hidden_states = False
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        capture_forward(
            model, input_ids, attention=True, hidden_states=True
        )
        assert model.config.output_attentions is False
        assert model.config.output_hidden_states is False

    def test_projection_spec_set(self):
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(model, input_ids, predictions=True)
        assert result.projection_spec is not None
        assert isinstance(result.projection_spec, ProjectionSpec)
        assert result.projection_spec.mode == "full"
        assert result.projection_spec.simplex_dim == VOCAB_SIZE

    def test_projection_spec_with_top_k(self):
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        policy = SamplingPolicy(top_k=10)
        result = capture_forward(
            model, input_ids, predictions=True, policy=policy
        )
        assert result.projection_spec.mode == "single_remainder"
        assert result.projection_spec.top_k == 10
        assert result.projection_spec.simplex_dim == 11

    def test_metadata_keys(self):
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(model, input_ids)
        assert "device" in result.metadata
        assert "seq_len" in result.metadata
        assert "vocab_size" in result.metadata
        assert "n_layers" in result.metadata

    def test_metadata_gqa_fields(self):
        """metadata should include n_heads, n_kv_heads, gqa_group_size."""
        model = _mock_model()
        model.config.num_key_value_heads = 2  # GQA: 4 Q-heads, 2 KV-heads
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(model, input_ids)
        assert result.metadata["n_heads"] == N_HEADS
        assert result.metadata["n_kv_heads"] == 2
        assert result.metadata["gqa_group_size"] == 2  # 4 // 2

    def test_metadata_gqa_no_kv_heads(self):
        """When num_key_value_heads is absent, n_kv_heads == n_heads (MHA)."""
        model = _mock_model()
        # Remove num_key_value_heads if set
        del model.config.num_key_value_heads
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(model, input_ids)
        assert result.metadata["n_heads"] == N_HEADS
        assert result.metadata["n_kv_heads"] == N_HEADS
        assert result.metadata["gqa_group_size"] == 1

    def test_nothing_requested(self):
        """If nothing requested, all fields are None."""
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(
            model,
            input_ids,
            predictions=False,
            attention=False,
            hidden_states=False,
            routing=False,
        )
        assert result.predictions is None
        assert result.attention is None
        assert result.hidden_states is None
        assert result.routing is None

    def test_projection_spec_known_tail(self):
        """known_tail mode: simplex_dim = K + tail_cardinality."""
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        policy = SamplingPolicy(
            top_k=10,
            remainder_mode="known_tail",
            tail_cardinality=90,
        )
        result = capture_forward(
            model, input_ids, predictions=True, policy=policy
        )
        assert result.projection_spec.mode == "known_tail"
        assert result.projection_spec.top_k == 10
        assert result.projection_spec.tail_cardinality == 90
        assert result.projection_spec.simplex_dim == 100  # 10 + 90

    def test_projection_spec_renormalize(self):
        """renormalize mode: simplex_dim = K."""
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        policy = SamplingPolicy(top_k=10, remainder_mode="renormalize")
        result = capture_forward(
            model, input_ids, predictions=True, policy=policy
        )
        assert result.projection_spec.mode == "renormalize"
        assert result.projection_spec.simplex_dim == 10

    def test_hidden_states_skip_embedding(self):
        """hidden_states[0] is the embedding; layer indices should match
        transformer layers, not include the embedding."""
        model = _mock_model(output_hidden_states=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(
            model, input_ids, predictions=False, hidden_states=True
        )
        # N_LAYERS transformer layers (not N_LAYERS + 1 including embedding)
        assert len(result.hidden_states) == N_LAYERS
        assert result.hidden_states[0]["layer"] == 0
        assert result.hidden_states[-1]["layer"] == N_LAYERS - 1

    def test_projection_spec_none_for_attention_only(self):
        """projection_spec should be None when only attention is extracted."""
        model = _mock_model(output_attentions=True)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(
            model,
            input_ids,
            predictions=False,
            attention=True,
            hidden_states=False,
        )
        assert result.attention is not None
        assert result.projection_spec is None

    def test_model_kwargs_passed_through(self):
        """Extra kwargs (e.g. attention_mask) should reach the model."""
        model = _mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

        # Replace __call__ with one that records kwargs
        received = {}

        def tracking_forward(input_ids, **kwargs):
            received.update(kwargs)
            return _mock_model_output()

        model.side_effect = tracking_forward
        model.__call__ = tracking_forward

        capture_forward(model, input_ids, attention_mask=mask)
        assert "attention_mask" in received
        assert torch.equal(received["attention_mask"], mask)

    def test_auto_moves_input_to_model_device(self):
        """input_ids should be auto-moved to the model's input device."""
        model = _mock_model()
        # Model embedding is on CPU; input_ids also on CPU.
        # Verify the forward call receives input_ids (no device mismatch).
        received_devices = []

        def tracking_forward(input_ids, **kwargs):
            received_devices.append(input_ids.device)
            return _mock_model_output()

        model.__call__ = tracking_forward
        model.side_effect = tracking_forward

        # Provide a real embedding so get_input_device can resolve
        embed = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        model.get_input_embeddings = MagicMock(return_value=embed)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        capture_forward(model, input_ids)
        assert received_devices[0] == torch.device("cpu")

    def test_get_output_embeddings_fallback(self):
        """When model has no lm_head, fall back to get_output_embeddings()."""
        model = _mock_model(output_hidden_states=True)
        # Move lm_head to get_output_embeddings and remove the attribute
        lm_head = model.lm_head
        del model.lm_head
        model.get_output_embeddings = MagicMock(return_value=lm_head)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(
            model, input_ids, predictions=False, hidden_states=True
        )
        assert result.hidden_states is not None
        model.get_output_embeddings.assert_called_once()

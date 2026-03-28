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

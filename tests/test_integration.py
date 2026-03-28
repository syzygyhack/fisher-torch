"""Integration tests for fisher_torch pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import torch

from fisher_torch.capture import CaptureResult, capture_forward
from fisher_torch.convert import from_simplex_array, to_simplex_array
from fisher_torch.extractors import (
    extract_attention,
    extract_predictions,
)

VOCAB_SIZE = 50
SEQ_LEN = 8
N_LAYERS = 3
N_HEADS = 2
HIDDEN_DIM = 32


class TestSyntheticLogitsPipeline:
    """Synthetic logits → extract_predictions → valid simplex."""

    def test_random_logits(self):
        torch.manual_seed(99)
        logits = torch.randn(SEQ_LEN, VOCAB_SIZE)
        result = extract_predictions(logits)
        assert result.dtype == np.float64
        assert result.shape == (SEQ_LEN, VOCAB_SIZE)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)
        assert (result >= 0).all()

    def test_extreme_logits(self):
        """Very large logits should not produce NaN/Inf."""
        logits = torch.tensor([[1000.0, -1000.0, 0.0]] * SEQ_LEN)
        result = extract_predictions(logits)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


class TestSyntheticAttentionPipeline:
    """Synthetic attention weights → extract_attention → valid simplex."""

    def test_softmax_attention(self):
        torch.manual_seed(100)
        layers = []
        for _ in range(N_LAYERS):
            raw = torch.randn(1, N_HEADS, SEQ_LEN, SEQ_LEN)
            layers.append(torch.softmax(raw, dim=-1))
        attn = tuple(layers)

        result = extract_attention(attn)
        assert isinstance(result, dict)
        for layer_idx, arr in result.items():
            assert isinstance(layer_idx, int)
            sums = arr.sum(axis=-1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-6)


class TestRoundtrip:
    """Roundtrip: numpy simplex → from_simplex_array → to_simplex_array."""

    def test_roundtrip_preserves_values(self):
        rng = np.random.default_rng(42)
        original = rng.dirichlet(np.ones(10), size=5).astype(np.float64)
        tensor = from_simplex_array(original, dtype=torch.float64)
        recovered = to_simplex_array(tensor)
        np.testing.assert_allclose(recovered, original, atol=1e-12)


class TestFullCaptureMock:
    """Full capture_forward with all outputs enabled."""

    def _mock_model(self):
        model = MagicMock()
        model.config = MagicMock()
        model.config.vocab_size = VOCAB_SIZE
        model.config.num_hidden_layers = N_LAYERS
        model.config.num_attention_heads = N_HEADS
        model.config.output_attentions = False
        model.config.output_hidden_states = False

        torch.manual_seed(42)
        model.lm_head = torch.nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
        type(model).device = PropertyMock(return_value=torch.device("cpu"))

        def forward_fn(input_ids, **kwargs):
            torch.manual_seed(42)
            output = MagicMock()
            output.logits = torch.randn(1, SEQ_LEN, VOCAB_SIZE)

            layers = []
            for _ in range(N_LAYERS):
                raw = torch.randn(1, N_HEADS, SEQ_LEN, SEQ_LEN)
                layers.append(torch.softmax(raw, dim=-1))
            output.attentions = tuple(layers)

            output.hidden_states = tuple(
                torch.randn(1, SEQ_LEN, HIDDEN_DIM)
                for _ in range(N_LAYERS + 1)
            )
            return output

        model.side_effect = forward_fn
        model.__call__ = forward_fn
        return model

    def test_all_outputs(self):
        model = self._mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(
            model,
            input_ids,
            predictions=True,
            attention=True,
            hidden_states=True,
        )
        assert isinstance(result, CaptureResult)
        assert result.predictions is not None
        assert result.attention is not None
        assert result.hidden_states is not None
        assert result.projection_spec is not None
        assert result.metadata != {}

    def test_predictions_valid_simplex(self):
        model = self._mock_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        result = capture_forward(model, input_ids, predictions=True)
        assert result.predictions.dtype == np.float64
        sums = result.predictions.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

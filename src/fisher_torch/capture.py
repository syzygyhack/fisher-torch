"""Forward-pass orchestrator for simplex extraction.

Runs a single model forward pass and extracts selected simplex-valued
outputs using the stateless extractor functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from fisher_torch.extractors import (
    extract_attention,
    extract_layerwise_predictions,
    extract_predictions,
    extract_routing,
)
from fisher_torch.sampling import SamplingPolicy


@dataclass(frozen=True)
class ProjectionSpec:
    """Describes how a simplex vector was constructed from model output.

    Two observations are projection-compatible iff they share
    identical ProjectionSpec values.
    """

    mode: str
    """Projection mode: ``"full"``, ``"single_remainder"``,
    ``"renormalized"``, or ``"known_tail"``."""

    top_k: int | None = None
    original_dim: int | None = None
    tail_cardinality: int | None = None
    simplex_dim: int = 0

    def is_compatible_with(self, other: ProjectionSpec) -> bool:
        """Return True if *other* has identical projection geometry."""
        return (
            self.mode == other.mode
            and self.simplex_dim == other.simplex_dim
            and self.top_k == other.top_k
            and self.tail_cardinality == other.tail_cardinality
        )


@dataclass
class CaptureResult:
    """Result of a :func:`capture_forward` call."""

    predictions: np.ndarray | None = None
    attention: dict[int, np.ndarray] | None = None
    hidden_states: list[dict] | None = None
    routing: np.ndarray | None = None
    projection_spec: ProjectionSpec | None = None
    metadata: dict = field(default_factory=dict)


def capture_forward(
    model,
    input_ids: Tensor,
    *,
    predictions: bool = True,
    attention: bool = False,
    hidden_states: bool = False,
    routing: bool = False,
    policy: SamplingPolicy | None = None,
) -> CaptureResult:
    """Run a single forward pass and extract simplex-valued outputs.

    Parameters
    ----------
    model
        A HuggingFace ``PreTrainedModel`` (or compatible mock).
    input_ids : Tensor
        Token IDs of shape ``(batch, seq_len)``.
    predictions : bool
        Extract prediction simplices from logits.
    attention : bool
        Extract attention weight simplices.
    hidden_states : bool
        Extract layerwise prediction simplices (logit lens).
    routing : bool
        Extract MoE routing simplices.
    policy : SamplingPolicy or None
        Controls extraction scope (layers, heads, positions, top-k).

    Returns
    -------
    CaptureResult
        Populated with requested extractions.
    """
    if policy is None:
        policy = SamplingPolicy()

    # Save original config.
    orig_output_attentions = getattr(model.config, "output_attentions", False)
    orig_output_hidden_states = getattr(
        model.config, "output_hidden_states", False
    )

    try:
        # Temporarily set config flags for the forward pass.
        if attention:
            model.config.output_attentions = True
        if hidden_states:
            model.config.output_hidden_states = True

        with torch.no_grad():
            outputs = model(
                input_ids,
                output_attentions=attention,
                output_hidden_states=hidden_states,
            )
    finally:
        # Always restore original config values.
        model.config.output_attentions = orig_output_attentions
        model.config.output_hidden_states = orig_output_hidden_states

    result = CaptureResult()

    # Extract predictions.
    if predictions and hasattr(outputs, "logits") and outputs.logits is not None:
        top_k = policy.top_k
        result.predictions = extract_predictions(
            outputs.logits,
            top_k=top_k,
            remainder_mode=policy.remainder_mode,
        )

    # Extract attention.
    if attention and hasattr(outputs, "attentions") and outputs.attentions is not None:
        result.attention = extract_attention(
            outputs.attentions, policy=policy
        )

    # Extract hidden states (logit lens).
    if (
        hidden_states
        and hasattr(outputs, "hidden_states")
        and outputs.hidden_states is not None
    ):
        result.hidden_states = extract_layerwise_predictions(
            outputs.hidden_states, model.lm_head, policy=policy
        )

    # Extract routing weights.
    if (
        routing
        and hasattr(outputs, "router_logits")
        and outputs.router_logits is not None
    ):
        result.routing = extract_routing(outputs.router_logits)

    # Build projection spec.
    vocab_size = getattr(model.config, "vocab_size", None)
    top_k = policy.top_k
    if top_k is not None:
        mode = policy.remainder_mode
        simplex_dim = top_k + 1 if mode == "single_remainder" else top_k
    else:
        mode = "full"
        simplex_dim = vocab_size or 0

    result.projection_spec = ProjectionSpec(
        mode=mode,
        top_k=top_k,
        original_dim=vocab_size,
        simplex_dim=simplex_dim,
    )

    # Build metadata.
    device = getattr(model, "device", torch.device("cpu"))
    seq_len = input_ids.shape[-1] if input_ids.ndim > 0 else 0
    n_layers = getattr(model.config, "num_hidden_layers", None)

    result.metadata = {
        "device": str(device),
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
        "model_class": type(model).__name__,
    }

    return result

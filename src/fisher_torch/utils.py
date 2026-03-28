"""Softmax and top-k simplex helpers.

Provides numerically stable softmax with temperature scaling and
top-k probability selection with multiple remainder-handling modes
via fisher-simplex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from fisher_simplex import topk_to_simplex as _fs_topk_to_simplex
from torch import Tensor

# Matches fisher-simplex convention for negligible probability mass.
_CLIP_EPSILON = 1e-30


@dataclass
class ProjectionSpec:
    """Describes a top-k simplex projection.

    Lightweight mirror of the canonical observatory ProjectionSpec.
    Records the parameters used so downstream code can interpret
    the resulting simplex array.
    """

    k: int
    remainder_mode: str
    tail_cardinality: int | None = None
    original_vocab_size: int | None = None


def safe_softmax(
    logits: Tensor, *, temperature: float = 1.0, dim: int = -1
) -> Tensor:
    """Numerically stable softmax with temperature scaling.

    Parameters
    ----------
    logits : Tensor
        Raw logit values, any shape.
    temperature : float, optional
        Temperature divisor applied before softmax. Default ``1.0``.
    dim : int, optional
        Dimension along which to compute softmax. Default ``-1``.

    Returns
    -------
    Tensor
        Probability distribution summing to 1 along *dim*.
    """
    scaled = logits / temperature
    # Subtract max for numerical stability (redundant with F.softmax but
    # ensures consistent behaviour across backends).
    scaled = scaled - scaled.max(dim=dim, keepdim=True).values
    probs = F.softmax(scaled, dim=dim)
    # Clip denormalized floats to exact zero.
    probs = torch.where(probs < _CLIP_EPSILON, torch.zeros_like(probs), probs)
    # Renormalize after clipping to maintain valid distribution.
    probs = probs / probs.sum(dim=dim, keepdim=True)
    return probs


def topk_softmax(
    logits: Tensor,
    k: int,
    *,
    temperature: float = 1.0,
    remainder_mode: str = "single_remainder",
    tail_cardinality: int | None = None,
) -> tuple[Tensor, ProjectionSpec]:
    """Apply softmax, select top-k, and project to simplex.

    Combines :func:`safe_softmax` with top-k selection and delegates
    remainder handling to :func:`fisher_simplex.topk_to_simplex`.

    Parameters
    ----------
    logits : Tensor
        Raw logit values of shape ``(..., V)`` where *V* is vocab size.
    k : int
        Number of top entries to retain.
    temperature : float, optional
        Temperature for softmax. Default ``1.0``.
    remainder_mode : {"single_remainder", "renormalize", "known_tail"}
        How to handle the tail mass.
    tail_cardinality : int or None
        Required when *remainder_mode* is ``"known_tail"``.

    Returns
    -------
    simplex : Tensor
        Simplex tensor. Shape depends on *remainder_mode*:
        ``(..., K+1)``, ``(..., K)``, or ``(..., K + tail_cardinality)``.
    spec : ProjectionSpec
        Metadata describing the projection applied.
    """
    probs = safe_softmax(logits, temperature=temperature, dim=-1)
    vocab_size = logits.shape[-1]

    # Select top-k values.
    top_vals, _ = torch.topk(probs, k, dim=-1)

    # Delegate to fisher-simplex for remainder handling.
    top_np = top_vals.detach().cpu().numpy().astype(np.float64)
    simplex_np = _fs_topk_to_simplex(
        top_np, mode=remainder_mode, tail_cardinality=tail_cardinality
    )

    # Convert back to tensor matching input device/dtype.
    simplex = torch.from_numpy(simplex_np.copy()).to(
        device=logits.device, dtype=logits.dtype
    )

    spec = ProjectionSpec(
        k=k,
        remainder_mode=remainder_mode,
        tail_cardinality=tail_cardinality,
        original_vocab_size=vocab_size,
    )
    return simplex, spec

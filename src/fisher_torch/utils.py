"""Softmax and top-k simplex helpers.

Provides numerically stable softmax with temperature scaling and
top-k probability selection with multiple remainder-handling modes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Matches fisher-simplex convention for negligible probability mass.
_CLIP_THRESHOLD = 1e-30

_VALID_MODES = {"single_remainder", "renormalize", "known_tail"}


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
    probs = F.softmax(scaled, dim=dim)
    # Clip denormalized floats to exact zero.
    probs = torch.where(probs < _CLIP_THRESHOLD, torch.zeros_like(probs), probs)
    # Renormalize after clipping to maintain valid distribution.
    probs = probs / probs.sum(dim=dim, keepdim=True)
    return probs


def topk_to_simplex(
    probs: Tensor,
    *,
    top_k: int,
    mode: str = "single_remainder",
    tail_cardinality: int | None = None,
) -> Tensor:
    """Select top-k probabilities and construct a simplex vector.

    Parameters
    ----------
    probs : Tensor
        Probability tensor of shape ``(..., V)`` where *V* is vocab size.
    top_k : int
        Number of top entries to retain.
    mode : {"single_remainder", "renormalize", "known_tail"}
        How to handle the tail mass (see spec Section 1.2).
    tail_cardinality : int or None
        Required when *mode* is ``"known_tail"``. Number of bins over
        which to spread the remainder mass uniformly.

    Returns
    -------
    Tensor
        Simplex vector of shape ``(..., K+1)``, ``(..., K)``, or
        ``(..., K + tail_cardinality)`` depending on *mode*.

    Raises
    ------
    ValueError
        If *mode* is invalid, or ``"known_tail"`` is used without
        *tail_cardinality*.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode {mode!r}. Expected one of {sorted(_VALID_MODES)}."
        )
    if mode == "known_tail" and tail_cardinality is None:
        raise ValueError("tail_cardinality is required when mode='known_tail'.")

    top_vals, _ = torch.topk(probs, top_k, dim=-1)
    top_sum = top_vals.sum(dim=-1, keepdim=True)

    if mode == "single_remainder":
        remainder = 1.0 - top_sum
        remainder = torch.clamp(remainder, min=0.0)
        return torch.cat([top_vals, remainder], dim=-1)

    if mode == "renormalize":
        return top_vals / top_sum

    # mode == "known_tail"
    remainder = 1.0 - top_sum
    remainder = torch.clamp(remainder, min=0.0)
    per_bin = remainder / tail_cardinality
    tail = per_bin.expand(*top_vals.shape[:-1], tail_cardinality)
    return torch.cat([top_vals, tail], dim=-1)

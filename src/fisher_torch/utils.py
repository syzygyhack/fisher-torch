"""Softmax, top-k simplex, device, and numpy simplex helpers.

Provides numerically stable softmax with temperature scaling, top-k
probability selection with multiple remainder-handling modes via
fisher-simplex, device resolution for ``device_map="auto"`` models,
and post-extraction simplex utilities (truncation, renormalization).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from fisher_simplex import topk_to_simplex as _fs_topk_to_simplex
from torch import Tensor


def get_input_device(model) -> torch.device:
    """Resolve the device that *model* expects for input tensors.

    Handles ``device_map="auto"`` models where different layers live on
    different devices — input tensors must be placed on the device of
    the embedding layer.

    Resolution order:

    1. ``model.get_input_embeddings()`` parameter device
    2. ``next(model.parameters())`` device
    3. ``torch.device("cpu")`` (fallback)

    Parameters
    ----------
    model
        A HuggingFace ``PreTrainedModel`` or compatible object.

    Returns
    -------
    torch.device
    """
    # Try HuggingFace's standard accessor first.
    get_embed = getattr(model, "get_input_embeddings", None)
    if get_embed is not None:
        try:
            return next(get_embed().parameters()).device
        except (StopIteration, AttributeError):
            pass

    # Fallback: first parameter in the model.
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def truncate_and_renormalize(
    simplices: np.ndarray,
    target_len: int,
    *,
    null_threshold: float = 1e-12,
) -> np.ndarray:
    """Truncate simplex vectors and renormalize for cross-sequence comparison.

    When comparing attention distributions across sequences of different
    lengths, truncate to a common length and renormalize so each row
    is a valid simplex over the shared support.

    Parameters
    ----------
    simplices : np.ndarray
        Simplex array of shape ``(..., seq_len)``.  The typical shape
        from multi-prompt extraction is
        ``(n_prompts, n_layers, n_heads, max_seq)``.
    target_len : int
        Desired length along the last axis.  Must be ``<= seq_len``.
    null_threshold : float, optional
        Rows whose truncated sum falls below this threshold are replaced
        with the uniform distribution.  Default ``1e-12``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(..., target_len)`` with rows summing to 1.
    """
    simplices = np.asarray(simplices, dtype=np.float64)
    current_len = simplices.shape[-1]
    if target_len > current_len:
        raise ValueError(
            f"target_len ({target_len}) exceeds array length ({current_len})."
        )
    if target_len == current_len:
        return simplices.copy()

    trunc = simplices[..., :target_len].copy()
    row_sums = trunc.sum(axis=-1, keepdims=True)

    # Replace null rows with uniform distribution.
    null_mask = row_sums.squeeze(-1) < null_threshold
    if null_mask.any():
        trunc[null_mask] = 1.0 / target_len
        row_sums[null_mask[..., np.newaxis]] = 1.0

    trunc /= row_sums
    return trunc


# Matches fisher-simplex convention for negligible probability mass.
_CLIP_EPSILON = 1e-30


@dataclass
class TopkResult:
    """Metadata for a top-k simplex projection.

    Records the parameters used by :func:`topk_softmax` so downstream
    code can interpret the resulting simplex array.  This is an internal
    helper — the canonical ``ProjectionSpec`` lives in ``capture.py``.
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
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}."
        )
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
) -> tuple[Tensor, TopkResult]:
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
    spec : TopkResult
        Metadata describing the projection applied.
    """
    probs = safe_softmax(logits, temperature=temperature, dim=-1)
    vocab_size = logits.shape[-1]

    if k > vocab_size:
        raise ValueError(
            f"top_k ({k}) exceeds vocabulary size ({vocab_size})."
        )

    if remainder_mode == "known_tail" and (
        tail_cardinality is None or tail_cardinality < 1
    ):
        raise ValueError(
            f"known_tail mode requires tail_cardinality >= 1, "
            f"got {tail_cardinality}."
        )

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

    result_meta = TopkResult(
        k=k,
        remainder_mode=remainder_mode,
        tail_cardinality=tail_cardinality,
        original_vocab_size=vocab_size,
    )
    return simplex, result_meta

"""Stateless extraction functions for model outputs.

Converts PyTorch model outputs (logits, attention weights, hidden states,
router logits) into validated numpy simplex arrays suitable for
fisher-simplex analysis.  Each function also supports ``return_tensors``
mode for gradient-enabled extraction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from fisher_torch.convert import to_simplex_array
from fisher_torch.sampling import SamplingPolicy
from fisher_torch.utils import safe_softmax, topk_softmax


def extract_predictions(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    remainder_mode: str = "single_remainder",
    tail_cardinality: int | None = None,
    return_tensors: bool = False,
) -> np.ndarray | Tensor:
    """Convert logits to a validated simplex array.

    Parameters
    ----------
    logits : Tensor
        Raw logits of shape ``(seq_len, vocab_size)`` or
        ``(batch, seq_len, vocab_size)``.  A leading batch dimension
        of size 1 is squeezed.
    temperature : float, optional
        Temperature divisor for softmax.  Default ``1.0``.
    top_k : int or None, optional
        If set, retain only the top-k probabilities plus a remainder bin.
    remainder_mode : str, optional
        How to handle tail mass when *top_k* is set.  Default
        ``"single_remainder"``.
    tail_cardinality : int or None, optional
        Number of tail bins for ``"known_tail"`` remainder mode.
    return_tensors : bool, optional
        If ``True``, return a torch ``Tensor`` instead of a numpy array.
        Incompatible with *top_k* (raises ``ValueError``).

    Returns
    -------
    np.ndarray or Tensor
        Float64 simplex array of shape ``(T, V)`` or ``(T, K+1)``,
        or a torch ``Tensor`` when ``return_tensors=True``.
    """
    if return_tensors and top_k is not None:
        raise ValueError(
            "top_k is not supported in gradient mode (return_tensors=True). "
            "Use full-vocabulary softmax by setting top_k=None."
        )

    # Squeeze batch=1 dimension.
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)

    if top_k is not None:
        simplex_t, _ = topk_softmax(
            logits,
            top_k,
            temperature=temperature,
            remainder_mode=remainder_mode,
            tail_cardinality=tail_cardinality,
        )
        return to_simplex_array(simplex_t)

    probs = safe_softmax(
        logits, temperature=temperature,
        differentiable=return_tensors,
    )
    if return_tensors:
        return probs
    return to_simplex_array(probs)


def _extract_attention_single(
    attention_weights: tuple[Tensor, ...],
    *,
    layers: list[int],
    heads: list[int],
    positions: list[int],
    causal: bool,
    out_len: int,
    return_tensors: bool,
) -> dict[int, np.ndarray] | dict[int, Tensor]:
    """Extract attention for a single batch element (3D tensors)."""
    result: dict = {}
    for layer_idx in layers:
        attn = attention_weights[layer_idx]
        # attn shape: (n_heads, seq_len, seq_len)
        rows = []
        for h in heads:
            for p in positions:
                if causal:
                    row = attn[h, p, : p + 1]
                    if row.shape[0] < out_len:
                        pad = torch.zeros(
                            out_len - row.shape[0],
                            device=row.device,
                            dtype=row.dtype,
                        )
                        row = torch.cat([row, pad])
                else:
                    row = attn[h, p, :]
                rows.append(row)

        if not rows:
            empty = torch.empty(0, out_len)
            if return_tensors:
                result[layer_idx] = empty
            else:
                result[layer_idx] = empty.numpy().astype(
                    np.float64
                )
            continue

        stacked = torch.stack(rows)  # (n_selected, out_len)
        if return_tensors:
            result[layer_idx] = stacked
        else:
            result[layer_idx] = to_simplex_array(stacked)

    return result


def extract_attention(
    attention_weights: Tensor | tuple[Tensor, ...],
    *,
    policy: SamplingPolicy | None = None,
    causal: bool = True,
    return_tensors: bool = False,
) -> (
    dict[int, np.ndarray]
    | list[dict[int, np.ndarray]]
    | dict[int, Tensor]
    | list[dict[int, Tensor]]
):
    """Convert attention weights to simplex arrays keyed by layer.

    Parameters
    ----------
    attention_weights : Tensor or tuple[Tensor, ...]
        Attention tensors from a model forward pass, one per layer.
        Each tensor has shape ``(batch, n_heads, seq_len, seq_len)``.
    policy : SamplingPolicy or None, optional
        Controls which layers, heads, and positions to extract.
        Defaults to ``SamplingPolicy()`` (final token only).
    causal : bool, optional
        If ``True`` (default), trim each attention row to its causal
        window ``[:position + 1]``.
    return_tensors : bool, optional
        If ``True``, return torch ``Tensor`` values instead of numpy.

    Returns
    -------
    dict or list[dict]
        For batch=1, a ``dict[int, ndarray|Tensor]`` keyed by layer.
        For batch>1, a ``list`` of such dicts, one per batch element.
    """
    if policy is None:
        policy = SamplingPolicy()

    if isinstance(attention_weights, Tensor):
        attention_weights = (attention_weights,)

    n_layers = len(attention_weights)
    n_heads = attention_weights[0].shape[1]
    seq_len = attention_weights[0].shape[-1]

    layers = policy.selected_layers(n_layers)
    heads = policy.selected_heads(n_heads)
    positions = policy.selected_positions(seq_len)

    if causal and positions:
        out_len = max(positions) + 1
    else:
        out_len = seq_len

    batch_size = attention_weights[0].shape[0]

    if batch_size == 1:
        # Squeeze batch dim from each layer tensor.
        squeezed = tuple(a.squeeze(0) for a in attention_weights)
        return _extract_attention_single(
            squeezed,
            layers=layers,
            heads=heads,
            positions=positions,
            causal=causal,
            out_len=out_len,
            return_tensors=return_tensors,
        )

    # Batch > 1: process each element independently.
    batch_results = []
    for b in range(batch_size):
        sliced = tuple(a[b] for a in attention_weights)
        batch_results.append(
            _extract_attention_single(
                sliced,
                layers=layers,
                heads=heads,
                positions=positions,
                causal=causal,
                out_len=out_len,
                return_tensors=return_tensors,
            )
        )
    return batch_results


def extract_layerwise_predictions(
    hidden_states: tuple[Tensor, ...],
    lm_head: nn.Module,
    *,
    policy: SamplingPolicy | None = None,
    no_grad: bool = True,
    return_tensors: bool = False,
) -> list[dict] | list[list[dict]]:
    """Project hidden states through the unembedding matrix (logit lens).

    Parameters
    ----------
    hidden_states : tuple[Tensor, ...]
        Hidden state tensors from each layer, shape
        ``(batch, seq_len, hidden_dim)``.
    lm_head : nn.Module
        The model's output projection (unembedding) layer.
    policy : SamplingPolicy or None, optional
        Controls which layers to extract and optional top-k.
        Defaults to ``SamplingPolicy()``.
    no_grad : bool, optional
        If ``True`` (default), wrap ``lm_head`` call in
        ``torch.no_grad()``.  Set ``False`` for gradient mode.
    return_tensors : bool, optional
        If ``True``, return torch ``Tensor`` values instead of numpy.
        Incompatible with *top_k* (raises ``ValueError``).

    Returns
    -------
    list[dict] or list[list[dict]]
        For batch=1, a list of dicts with ``"layer"`` and
        ``"predictions"`` keys.  For batch>1, a list of such lists.
    """
    if return_tensors and policy is not None and policy.top_k is not None:
        raise ValueError(
            "top_k is not supported in gradient mode (return_tensors=True). "
            "Use full-vocabulary softmax by setting top_k=None."
        )

    if policy is None:
        policy = SamplingPolicy()

    n_layers = len(hidden_states)
    layers = policy.selected_layers(n_layers)
    batch_size = hidden_states[0].shape[0] if hidden_states[0].ndim == 3 else 1

    if batch_size == 1:
        return _extract_layerwise_single(
            hidden_states, lm_head, layers=layers, policy=policy,
            no_grad=no_grad, return_tensors=return_tensors,
        )

    batch_results = []
    for b in range(batch_size):
        sliced = tuple(hs[b : b + 1] for hs in hidden_states)
        batch_results.append(
            _extract_layerwise_single(
                sliced, lm_head, layers=layers, policy=policy,
                no_grad=no_grad, return_tensors=return_tensors,
            )
        )
    return batch_results


def _extract_layerwise_single(
    hidden_states: tuple[Tensor, ...],
    lm_head: nn.Module,
    *,
    layers: list[int],
    policy: SamplingPolicy,
    no_grad: bool,
    return_tensors: bool,
) -> list[dict]:
    """Extract layerwise predictions for a single batch element."""
    results = []
    for layer_idx in layers:
        hs = hidden_states[layer_idx]
        if hs.ndim == 3 and hs.shape[0] == 1:
            hs = hs.squeeze(0)

        positions = policy.selected_positions(hs.shape[0])
        hs = hs[positions]

        if no_grad:
            with torch.no_grad():
                layer_logits = lm_head(hs)
        else:
            layer_logits = lm_head(hs)

        if policy.top_k is not None:
            simplex_t, _ = topk_softmax(
                layer_logits,
                policy.top_k,
                remainder_mode=policy.remainder_mode,
                tail_cardinality=policy.tail_cardinality,
            )
            predictions = to_simplex_array(simplex_t)
        else:
            probs = safe_softmax(
                layer_logits,
                differentiable=return_tensors,
            )
            if return_tensors:
                predictions = probs
            else:
                predictions = to_simplex_array(probs)

        results.append({
            "layer": layer_idx,
            "predictions": predictions,
        })

    return results


def extract_hidden_states(
    hidden_states: tuple[Tensor, ...],
    *,
    policy: SamplingPolicy | None = None,
    return_tensors: bool = False,
) -> list[dict] | list[list[dict]]:
    """Extract raw hidden state vectors without lm_head projection.

    Parameters
    ----------
    hidden_states : tuple[Tensor, ...]
        Hidden state tensors from each layer, shape
        ``(batch, seq_len, hidden_dim)``.
    policy : SamplingPolicy or None, optional
        Controls which layers and positions to extract.
        Defaults to ``SamplingPolicy()``.
    return_tensors : bool, optional
        If ``True``, return torch ``Tensor`` values instead of numpy.

    Returns
    -------
    list[dict] or list[list[dict]]
        For batch=1, a list of dicts with ``"layer"`` (int) and
        ``"hidden_states"`` (ndarray of shape ``(n_positions, hidden_dim)``
        or ``Tensor``).  For batch>1, a list of such lists.
    """
    if policy is None:
        policy = SamplingPolicy()

    n_layers = len(hidden_states)
    layers = policy.selected_layers(n_layers)
    batch_size = hidden_states[0].shape[0] if hidden_states[0].ndim == 3 else 1

    if batch_size == 1:
        return _extract_hidden_single(
            hidden_states, layers=layers, policy=policy,
            return_tensors=return_tensors,
        )

    batch_results = []
    for b in range(batch_size):
        sliced = tuple(hs[b : b + 1] for hs in hidden_states)
        batch_results.append(
            _extract_hidden_single(
                sliced, layers=layers, policy=policy,
                return_tensors=return_tensors,
            )
        )
    return batch_results


def _extract_hidden_single(
    hidden_states: tuple[Tensor, ...],
    *,
    layers: list[int],
    policy: SamplingPolicy,
    return_tensors: bool,
) -> list[dict]:
    """Extract raw hidden states for a single batch element."""
    results = []
    for layer_idx in layers:
        hs = hidden_states[layer_idx]
        if hs.ndim == 3 and hs.shape[0] == 1:
            hs = hs.squeeze(0)

        positions = policy.selected_positions(hs.shape[0])
        selected = hs[positions]  # (n_positions, hidden_dim)

        if return_tensors:
            vectors = selected
        else:
            vectors = selected.detach().cpu().numpy().astype(np.float64)

        results.append({
            "layer": layer_idx,
            "hidden_states": vectors,
        })

    return results


def extract_routing(
    router_logits: Tensor | tuple[Tensor, ...],
    *,
    return_tensors: bool = False,
) -> np.ndarray | Tensor:
    """Convert MoE router logits to a validated simplex array.

    Parameters
    ----------
    router_logits : Tensor or tuple[Tensor, ...]
        Router logits of shape ``(tokens, n_experts)``.  If a tuple,
        tensors are concatenated along the token dimension.
    return_tensors : bool, optional
        If ``True``, return a torch ``Tensor`` instead of a numpy array.

    Returns
    -------
    np.ndarray or Tensor
        Float64 simplex array of shape ``(tokens, n_experts)``,
        or a torch ``Tensor`` when ``return_tensors=True``.
    """
    if isinstance(router_logits, tuple):
        router_logits = torch.cat(router_logits, dim=0)

    probs = safe_softmax(
        router_logits, differentiable=return_tensors,
    )
    if return_tensors:
        return probs
    return to_simplex_array(probs)

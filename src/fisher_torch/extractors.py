"""Stateless extraction functions for model outputs.

Converts PyTorch model outputs (logits, attention weights, hidden states,
router logits) into validated numpy simplex arrays suitable for
fisher-simplex analysis.
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
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Float64 simplex array of shape ``(T, V)`` or ``(T, K+1)``.
    """
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

    probs = safe_softmax(logits, temperature=temperature)
    return to_simplex_array(probs)


def extract_attention(
    attention_weights: Tensor | tuple[Tensor, ...],
    *,
    policy: SamplingPolicy | None = None,
    causal: bool = True,
) -> dict[int, np.ndarray]:
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
        window ``[:position + 1]``.  This removes the zero-padding
        from the causal mask, producing a properly-sized simplex.

        When all selected positions share the same causal length
        (e.g. ``final_token_only``), the output shape is unchanged.
        When positions have different causal lengths, rows are padded
        to the length of the longest selected position.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from layer index to float64 simplex array of shape
        ``(n_selected_heads * n_selected_positions, causal_len)``
        where *causal_len* is ``max(positions) + 1`` when
        ``causal=True``, or ``seq_len`` when ``causal=False``.
    """
    if policy is None:
        policy = SamplingPolicy()

    if isinstance(attention_weights, Tensor):
        attention_weights = (attention_weights,)

    n_layers = len(attention_weights)
    # Infer head count and seq_len from first tensor.
    n_heads = attention_weights[0].shape[1]
    seq_len = attention_weights[0].shape[-1]

    layers = policy.selected_layers(n_layers)
    heads = policy.selected_heads(n_heads)
    positions = policy.selected_positions(seq_len)

    # Determine output row length.
    if causal and positions:
        out_len = max(positions) + 1
    else:
        out_len = seq_len

    result: dict[int, np.ndarray] = {}
    for layer_idx in layers:
        attn = attention_weights[layer_idx]
        # attn shape: (batch, n_heads, seq_len, seq_len)
        if attn.ndim == 4 and attn.shape[0] == 1:
            attn = attn.squeeze(0)  # (n_heads, seq_len, seq_len)
        elif attn.ndim == 4 and attn.shape[0] > 1:
            raise ValueError(
                f"Batch size {attn.shape[0]} > 1 is not supported for "
                "attention extraction."
            )

        rows = []
        for h in heads:
            for p in positions:
                if causal:
                    row = attn[h, p, : p + 1]
                    # Pad to out_len if needed (multi-position case).
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

        stacked = torch.stack(rows)  # (n_selected, out_len)
        result[layer_idx] = to_simplex_array(stacked)

    return result


def extract_layerwise_predictions(
    hidden_states: tuple[Tensor, ...],
    lm_head: nn.Module,
    *,
    policy: SamplingPolicy | None = None,
) -> list[dict]:
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

    Returns
    -------
    list[dict]
        Each dict has ``"layer"`` (int) and ``"predictions"`` (ndarray).
    """
    if policy is None:
        policy = SamplingPolicy()

    n_layers = len(hidden_states)
    layers = policy.selected_layers(n_layers)

    results = []
    for layer_idx in layers:
        hs = hidden_states[layer_idx]
        if hs.ndim == 3 and hs.shape[0] == 1:
            hs = hs.squeeze(0)  # (seq_len, hidden_dim)
        elif hs.ndim == 3 and hs.shape[0] > 1:
            raise ValueError(
                f"Batch size {hs.shape[0]} > 1 is not supported for "
                "layerwise prediction extraction."
            )

        # Select positions before projecting through lm_head.
        positions = policy.selected_positions(hs.shape[0])
        hs = hs[positions]  # (n_positions, hidden_dim)

        with torch.no_grad():
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
            probs = safe_softmax(layer_logits)
            predictions = to_simplex_array(probs)

        results.append({
            "layer": layer_idx,
            "predictions": predictions,
        })

    return results


def extract_routing(
    router_logits: Tensor | tuple[Tensor, ...],
) -> np.ndarray:
    """Convert MoE router logits to a validated simplex array.

    Parameters
    ----------
    router_logits : Tensor or tuple[Tensor, ...]
        Router logits of shape ``(tokens, n_experts)``.  If a tuple,
        tensors are concatenated along the token dimension.

    Returns
    -------
    np.ndarray
        Float64 simplex array of shape ``(tokens, n_experts)``.
    """
    if isinstance(router_logits, tuple):
        router_logits = torch.cat(router_logits, dim=0)

    probs = safe_softmax(router_logits)
    return to_simplex_array(probs)

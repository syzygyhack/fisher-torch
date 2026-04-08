"""Tensor-simplex conversion utilities.

Converts between PyTorch tensors and validated numpy simplex arrays
suitable for fisher-simplex operations.
"""

from __future__ import annotations

import numpy as np
import torch
from fisher_simplex.utils import validate_simplex
from numpy.typing import NDArray
from torch import Tensor


def to_simplex_array(tensor: Tensor, *, dim: int = -1) -> NDArray[np.floating]:
    """Detach a torch tensor and return a validated float64 simplex array.

    Parameters
    ----------
    tensor : Tensor
        Probability tensor of shape ``(N,)`` or ``(M, N)``.
    dim : int, optional
        Axis along which components sum to 1. Default ``-1``.

    Returns
    -------
    NDArray[np.floating]
        Validated simplex array in float64.
    """
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    return validate_simplex(arr, axis=dim, renormalize="warn")


def stack_attention(
    attention: dict[int, np.ndarray],
    *,
    n_heads: int,
) -> np.ndarray:
    """Stack a per-layer attention dict into a dense array.

    Converts the ``dict[int, np.ndarray]`` returned by
    :func:`~fisher_torch.extractors.extract_attention` into a single
    array suitable for vectorised geometry operations.

    Parameters
    ----------
    attention : dict[int, np.ndarray]
        Mapping from layer index to array of shape
        ``(n_heads * n_positions, seq_len)``.
    n_heads : int
        Number of attention heads (used to reshape rows).

    Returns
    -------
    np.ndarray
        If each layer has a single query position (the default
        ``final_token_only`` mode), returns shape
        ``(n_layers, n_heads, seq_len)``.

        Otherwise returns ``(n_layers, n_heads, n_positions, seq_len)``.
    """
    layers = sorted(attention.keys())
    arrays = []
    for layer_idx in layers:
        arr = attention[layer_idx]  # (n_heads * n_positions, seq_len)
        n_rows, seq_len = arr.shape
        n_positions = n_rows // n_heads
        arrays.append(arr.reshape(n_heads, n_positions, seq_len))

    stacked = np.stack(arrays, axis=0)  # (n_layers, n_heads, n_positions, seq_len)

    # Squeeze single-position dimension for the common final-token case.
    if stacked.shape[2] == 1:
        stacked = stacked.squeeze(2)

    return stacked


def from_simplex_array(
    array: np.ndarray,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Convert a validated numpy simplex array to a torch tensor.

    Parameters
    ----------
    array : np.ndarray
        Simplex array of shape ``(N,)`` or ``(M, N)``.
    device : torch.device or str, optional
        Target device. Default CPU.
    dtype : torch.dtype, optional
        Target dtype. Default ``torch.float32``.

    Returns
    -------
    Tensor
        Tensor on the requested device and dtype.
    """
    validated = validate_simplex(array, renormalize="warn")
    t = torch.from_numpy(validated.copy())
    if dtype is None:
        dtype = torch.float32
    return t.to(device=device, dtype=dtype)

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

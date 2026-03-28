"""Sampling policy for controlling extraction scope.

Defines which layers, heads, and token positions to extract from
a forward pass, preventing extraction volume from exploding on
large models.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingPolicy:
    """Controls which parts of a forward pass to extract.

    v0.1 defaults: final-token prediction geometry only.
    This prevents extraction from exploding in volume on large models.
    """

    layers: list[int] | None = None
    heads: list[int] | None = None
    positions: list[int] | slice | None = None
    final_token_only: bool = True
    layer_stride: int | None = None
    head_stride: int | None = None
    max_tokens_per_sample: int | None = None
    top_k: int | None = None
    remainder_mode: str = "single_remainder"

    def selected_layers(self, n_layers: int) -> list[int]:
        """Return layer indices to extract.

        Parameters
        ----------
        n_layers : int
            Total number of layers in the model.

        Returns
        -------
        list[int]
            Indices of layers to extract from.

        Raises
        ------
        ValueError
            If any index is out of bounds.
        """
        if self.layers is not None:
            _validate_bounds(self.layers, n_layers, "layer")
            return list(self.layers)
        if self.layer_stride is not None:
            return list(range(0, n_layers, self.layer_stride))
        return list(range(n_layers))

    def selected_heads(self, n_heads: int) -> list[int]:
        """Return head indices to extract.

        Parameters
        ----------
        n_heads : int
            Total number of attention heads.

        Returns
        -------
        list[int]
            Indices of heads to extract from.

        Raises
        ------
        ValueError
            If any index is out of bounds.
        """
        if self.heads is not None:
            _validate_bounds(self.heads, n_heads, "head")
            return list(self.heads)
        if self.head_stride is not None:
            return list(range(0, n_heads, self.head_stride))
        return list(range(n_heads))

    def selected_positions(self, seq_len: int) -> list[int]:
        """Return token position indices to extract.

        Parameters
        ----------
        seq_len : int
            Sequence length of the current input.

        Returns
        -------
        list[int]
            Indices of positions to extract from.

        Raises
        ------
        ValueError
            If any index is out of bounds.
        """
        if self.final_token_only:
            return [seq_len - 1]

        if self.positions is not None:
            if isinstance(self.positions, slice):
                result = list(range(seq_len))[self.positions]
            else:
                _validate_bounds(self.positions, seq_len, "position")
                result = list(self.positions)
        else:
            result = list(range(seq_len))

        if self.max_tokens_per_sample is not None:
            result = result[: self.max_tokens_per_sample]
        return result


def _validate_bounds(indices: list[int], upper: int, name: str) -> None:
    """Raise ValueError if any index is out of [0, upper)."""
    for idx in indices:
        if idx < 0 or idx >= upper:
            raise ValueError(
                f"{name} index {idx} out of bounds for size {upper}."
            )

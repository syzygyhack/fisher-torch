"""Sampling policy for controlling extraction scope.

Defines which layers, heads, and token positions to extract from
a forward pass, preventing extraction volume from exploding on
large models.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_PRESETS = ("atlas", "quartiles")


@dataclass
class SamplingPolicy:
    """Controls which parts of a forward pass to extract.

    Defaults to final-token prediction geometry only, preventing
    extraction volume from exploding on large models.
    """

    layers: list[int] | None = None
    heads: list[int] | None = None
    positions: list[int] | slice | None = None
    final_token_only: bool = True
    position_preset: str | None = None
    layer_stride: int | None = None
    head_stride: int | None = None
    max_tokens_per_sample: int | None = None
    top_k: int | None = None
    remainder_mode: str = "single_remainder"
    tail_cardinality: int | None = None

    def __post_init__(self) -> None:
        if self.position_preset is not None:
            if self.position_preset not in _VALID_PRESETS:
                raise ValueError(
                    f"Unknown position_preset {self.position_preset!r}. "
                    f"Valid presets: {_VALID_PRESETS}"
                )

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

        When *position_preset* is set, positions are computed dynamically
        from *seq_len* and the preset overrides both *final_token_only*
        and *positions*.

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
        if seq_len <= 0:
            return []

        # Preset overrides final_token_only and explicit positions.
        if self.position_preset is not None:
            labeled = _resolve_preset(self.position_preset, seq_len)
            result = list(labeled.values())
            if self.max_tokens_per_sample is not None:
                result = result[: self.max_tokens_per_sample]
            return result

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

    def position_labels(self, seq_len: int) -> dict[str, int] | None:
        """Return labeled positions when a preset is active.

        The returned labels match the positions returned by
        :meth:`selected_positions`, including any truncation from
        *max_tokens_per_sample*.

        Parameters
        ----------
        seq_len : int
            Sequence length of the current input.

        Returns
        -------
        dict[str, int] or None
            Mapping from label to position index when a preset is set,
            ``None`` otherwise.
        """
        if self.position_preset is None or seq_len <= 0:
            return None
        labeled = _resolve_preset(self.position_preset, seq_len)
        if self.max_tokens_per_sample is not None:
            labeled = dict(
                list(labeled.items())[: self.max_tokens_per_sample]
            )
        return labeled


def _resolve_preset(preset: str, seq_len: int) -> dict[str, int]:
    """Compute named positions for a preset.

    Returns an ordered dict (insertion-ordered) of label -> position.
    Positions are deduplicated; if two labels map to the same index,
    only the first is kept.
    """
    if preset == "atlas":
        candidates = [
            ("early", min(4, seq_len - 1)),
            ("mid", seq_len // 2),
            ("late", max(seq_len - 3, seq_len // 2 + 1)),
            ("final", seq_len - 1),
        ]
    elif preset == "quartiles":
        candidates = [
            ("q25", seq_len // 4),
            ("q50", seq_len // 2),
            ("q75", 3 * seq_len // 4),
            ("q100", seq_len - 1),
        ]
    else:
        raise ValueError(
            f"Unknown position_preset {preset!r}. "
            f"Valid presets: {_VALID_PRESETS}"
        )

    # Deduplicate by position, keeping first label.  Clamp to valid range.
    seen: set[int] = set()
    result: dict[str, int] = {}
    for label, pos in candidates:
        pos = max(0, min(pos, seq_len - 1))
        if pos not in seen:
            seen.add(pos)
            result[label] = pos
    return result


def _validate_bounds(indices: list[int], upper: int, name: str) -> None:
    """Raise ValueError if any index is out of [0, upper)."""
    for idx in indices:
        if idx < 0 or idx >= upper:
            raise ValueError(
                f"{name} index {idx} out of bounds for size {upper}."
            )

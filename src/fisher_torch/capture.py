"""Forward-pass orchestrator for simplex extraction.

Runs a single model forward pass and extracts selected simplex-valued
outputs using the stateless extractor functions.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from fisher_torch.convert import to_simplex_array
from fisher_torch.extractors import (
    extract_attention,
    extract_hidden_states,
    extract_layerwise_predictions,
    extract_predictions,
    extract_routing,
)
from fisher_torch.sampling import SamplingPolicy
from fisher_torch.utils import get_input_device


@dataclass(frozen=True)
class ProjectionSpec:
    """Describes how a simplex vector was constructed from model output.

    Two observations are projection-compatible iff they share
    identical ProjectionSpec values.
    """

    mode: str
    """Projection mode: ``"full"``, ``"single_remainder"``,
    ``"renormalize"``, or ``"known_tail"``."""

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
    """Result of a :func:`capture_forward` call.

    **Float64 guarantee (numpy path):** When ``no_grad=True`` (the
    default), all numpy arrays — ``predictions``, ``attention`` values,
    ``hidden_states`` prediction entries, and ``routing`` — are
    ``float64``.  This is enforced by :func:`~fisher_torch.convert.to_simplex_array`
    and is required for geometric stability in downstream
    fisher-simplex operations.  The ``raw_hidden_states`` vectors are
    also ``float64`` (cast explicitly).

    When ``no_grad=False``, the ``*_tensors`` fields hold torch
    ``Tensor`` objects with the model's native dtype (typically
    ``float32``).  Call :meth:`detach_to_numpy` to convert them to
    ``float64`` numpy arrays.
    """

    # Numpy simplex fields (populated in default no_grad=True mode).
    # All arrays are guaranteed float64.
    predictions: np.ndarray | None = None
    attention: dict[int, np.ndarray] | list[dict[int, np.ndarray]] | None = None
    hidden_states: list[dict] | list[list[dict]] | None = None
    raw_hidden_states: list[dict] | list[list[dict]] | None = None
    routing: np.ndarray | None = None

    # Tensor fields (populated when no_grad=False).
    prediction_tensors: Tensor | None = None
    attention_tensors: dict[int, Tensor] | list[dict[int, Tensor]] | None = None
    hidden_state_tensors: list[dict] | list[list[dict]] | None = None
    raw_hidden_state_tensors: list[dict] | list[list[dict]] | None = None
    routing_tensors: Tensor | None = None

    projection_spec: ProjectionSpec | None = None
    gqa_groups: list[int] | None = None
    metadata: dict = field(default_factory=dict)

    def detach_to_numpy(self) -> None:
        """Convert tensor fields to numpy and clear tensor references.

        Populates the numpy fields from their tensor counterparts,
        then sets the tensor fields to ``None``.
        """
        if self.prediction_tensors is not None:
            self.predictions = to_simplex_array(self.prediction_tensors)
            self.prediction_tensors = None

        if self.attention_tensors is not None:
            if isinstance(self.attention_tensors, list):
                self.attention = [
                    {k: to_simplex_array(v) for k, v in d.items()}
                    for d in self.attention_tensors
                ]
            else:
                self.attention = {
                    k: to_simplex_array(v)
                    for k, v in self.attention_tensors.items()
                }
            self.attention_tensors = None

        if self.hidden_state_tensors is not None:
            self.hidden_states = _detach_layer_dicts(
                self.hidden_state_tensors, "predictions"
            )
            self.hidden_state_tensors = None

        if self.raw_hidden_state_tensors is not None:
            self.raw_hidden_states = _detach_layer_dicts(
                self.raw_hidden_state_tensors, "hidden_states",
                simplex=False,
            )
            self.raw_hidden_state_tensors = None

        if self.routing_tensors is not None:
            self.routing = to_simplex_array(self.routing_tensors)
            self.routing_tensors = None


def _detach_layer_dicts(
    data: list[dict] | list[list[dict]],
    value_key: str,
    *,
    simplex: bool = True,
) -> list[dict] | list[list[dict]]:
    """Convert tensor values in layer-dict structures to numpy."""
    if not data:
        return data
    if isinstance(data[0], list):
        # Batched: list[list[dict]]
        return [
            _detach_layer_dicts(batch, value_key, simplex=simplex)
            for batch in data
        ]
    result = []
    for entry in data:
        t = entry[value_key]
        if simplex:
            arr = to_simplex_array(t)
        else:
            arr = t.detach().cpu().numpy().astype(np.float64)
        result.append({**entry, value_key: arr})
    return result


def capture_forward(
    model,
    input_ids: Tensor,
    *,
    predictions: bool = True,
    attention: bool = False,
    hidden_states: bool = False,
    raw_hidden_states: bool = False,
    routing: bool = False,
    policy: SamplingPolicy | None = None,
    no_grad: bool = True,
    **model_kwargs,
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
    raw_hidden_states : bool
        Extract raw hidden state vectors (no lm_head projection).
    routing : bool
        Extract MoE routing simplices.
    policy : SamplingPolicy or None
        Controls extraction scope (layers, heads, positions, top-k).
    no_grad : bool
        If ``True`` (default), wrap the forward pass in
        ``torch.no_grad()`` and return numpy arrays.  If ``False``,
        preserve the computation graph and populate the ``*_tensors``
        fields on :class:`CaptureResult` instead.
    **model_kwargs
        Additional keyword arguments forwarded to the model forward call
        (e.g. ``attention_mask``, ``position_ids``).

    Returns
    -------
    CaptureResult
        Populated with requested extractions.  When ``no_grad=True``
        (the default), all numpy arrays are guaranteed ``float64``.
        When ``no_grad=False``, the ``*_tensors`` fields hold torch
        ``Tensor`` objects in the model's native dtype; call
        :meth:`CaptureResult.detach_to_numpy` to convert to ``float64``.
    """
    if policy is None:
        policy = SamplingPolicy()

    return_tensors = not no_grad

    # Auto-move input tensors to the model's input device (handles device_map="auto").
    target_device = get_input_device(model)
    if input_ids.device != target_device:
        input_ids = input_ids.to(target_device)
    for key, val in model_kwargs.items():
        if isinstance(val, Tensor) and val.device != target_device:
            model_kwargs[key] = val.to(target_device)

    # Guard: gradient checkpointing + output_attentions is incompatible.
    if attention and getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError(
            "Cannot extract attention weights when gradient checkpointing "
            "is enabled. Checkpointed layers do not save intermediate "
            "activations (including attention weights). Disable gradient "
            "checkpointing with model.gradient_checkpointing_disable() "
            "before calling capture_forward with attention=True."
        )

    need_hidden = hidden_states or raw_hidden_states

    # Save original config.
    orig_output_attentions = getattr(model.config, "output_attentions", False)
    orig_output_hidden_states = getattr(
        model.config, "output_hidden_states", False
    )

    try:
        # Temporarily set config flags for the forward pass.
        if attention:
            model.config.output_attentions = True
        if need_hidden:
            model.config.output_hidden_states = True

        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            outputs = model(
                input_ids,
                output_attentions=attention,
                output_hidden_states=need_hidden,
                **model_kwargs,
            )
    finally:
        # Always restore original config values.
        model.config.output_attentions = orig_output_attentions
        model.config.output_hidden_states = orig_output_hidden_states

    result = CaptureResult()

    # Extract predictions.
    if predictions and hasattr(outputs, "logits") and outputs.logits is not None:
        extracted = extract_predictions(
            outputs.logits,
            top_k=policy.top_k,
            remainder_mode=policy.remainder_mode,
            tail_cardinality=policy.tail_cardinality,
            return_tensors=return_tensors,
        )
        if return_tensors:
            result.prediction_tensors = extracted
        else:
            result.predictions = extracted

    # Extract attention.
    if attention and hasattr(outputs, "attentions") and outputs.attentions is not None:
        extracted = extract_attention(
            outputs.attentions, policy=policy,
            return_tensors=return_tensors,
        )
        if return_tensors:
            result.attention_tensors = extracted
        else:
            result.attention = extracted

    # Hidden state tuple (shared by logit-lens and raw extraction).
    hs_tuple = None
    if (
        need_hidden
        and hasattr(outputs, "hidden_states")
        and outputs.hidden_states is not None
    ):
        # HuggingFace includes embedding at index 0; skip it.
        hs_tuple = outputs.hidden_states[1:]

    # Extract hidden states (logit lens).
    if hidden_states and hs_tuple is not None:
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None and hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings()
        extracted = extract_layerwise_predictions(
            hs_tuple, lm_head, policy=policy,
            no_grad=no_grad, return_tensors=return_tensors,
        )
        if return_tensors:
            result.hidden_state_tensors = extracted
        else:
            result.hidden_states = extracted

    # Extract raw hidden states.
    if raw_hidden_states and hs_tuple is not None:
        extracted = extract_hidden_states(
            hs_tuple, policy=policy,
            return_tensors=return_tensors,
        )
        if return_tensors:
            result.raw_hidden_state_tensors = extracted
        else:
            result.raw_hidden_states = extracted

    # Extract routing weights.
    if (
        routing
        and hasattr(outputs, "router_logits")
        and outputs.router_logits is not None
    ):
        extracted = extract_routing(
            outputs.router_logits, return_tensors=return_tensors,
        )
        if return_tensors:
            result.routing_tensors = extracted
        else:
            result.routing = extracted

    # Build projection spec (only meaningful when simplex data is extracted).
    has_predictions = (
        result.predictions is not None or result.prediction_tensors is not None
    )
    has_hidden = (
        result.hidden_states is not None
        or result.hidden_state_tensors is not None
    )
    if has_predictions or has_hidden:
        vocab_size = getattr(model.config, "vocab_size", None)
        top_k = policy.top_k
        tail_cardinality = policy.tail_cardinality
        if top_k is not None:
            mode = policy.remainder_mode
            if mode == "single_remainder":
                simplex_dim = top_k + 1
            elif mode == "known_tail" and tail_cardinality is not None:
                simplex_dim = top_k + tail_cardinality
            else:
                simplex_dim = top_k
        else:
            mode = "full"
            simplex_dim = vocab_size or 0

        result.projection_spec = ProjectionSpec(
            mode=mode,
            top_k=top_k,
            original_dim=vocab_size,
            tail_cardinality=tail_cardinality,
            simplex_dim=simplex_dim,
        )

    # Build metadata.
    device = getattr(model, "device", torch.device("cpu"))
    seq_len = input_ids.shape[-1] if input_ids.ndim > 0 else 0
    n_layers = getattr(model.config, "num_hidden_layers", None)
    meta_vocab_size = getattr(model.config, "vocab_size", None)
    n_heads = getattr(model.config, "num_attention_heads", None)
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    gqa_group_size = (
        n_heads // n_kv_heads if n_heads and n_kv_heads else 1
    )

    # Build per-head GQA group mapping aligned to extracted heads.
    # Assumes uniform GQA (n_heads divisible by n_kv_heads).
    if n_heads and n_kv_heads:
        if n_heads % n_kv_heads != 0:
            result.gqa_groups = None
        else:
            selected_heads = policy.selected_heads(n_heads)
            result.gqa_groups = [
                h // gqa_group_size for h in selected_heads
            ]
    else:
        result.gqa_groups = None

    result.metadata = {
        "device": str(device),
        "seq_len": seq_len,
        "vocab_size": meta_vocab_size,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "gqa_group_size": gqa_group_size,
        "model_class": type(model).__name__,
    }

    return result


@dataclass
class BatchCaptureResult:
    """Result of a :func:`capture_batch` call."""

    results: list[CaptureResult]
    aligned_attention: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


def capture_batch(
    model,
    prompts: list[Tensor],
    *,
    predictions: bool = True,
    attention: bool = False,
    hidden_states: bool = False,
    raw_hidden_states: bool = False,
    routing: bool = False,
    policy: SamplingPolicy | None = None,
    no_grad: bool = True,
    **model_kwargs,
) -> BatchCaptureResult:
    """Run multiple variable-length prompts and align results.

    Each prompt is run through :func:`capture_forward` independently.
    When *attention* is requested, attention arrays are padded to a
    common shape along the sequence-length axis.

    Parameters
    ----------
    model
        A HuggingFace ``PreTrainedModel`` (or compatible mock).
    prompts : list[Tensor]
        List of token ID tensors.  Each may be 1D ``(seq_len,)`` or
        2D ``(1, seq_len)``.
    predictions, attention, hidden_states, raw_hidden_states, routing
        Which outputs to extract (forwarded to :func:`capture_forward`).
    policy : SamplingPolicy or None
        Shared extraction policy for all prompts.
    no_grad : bool
        Gradient mode flag (forwarded to :func:`capture_forward`).
    **model_kwargs
        Extra kwargs forwarded to every forward call.  These are
        shared across all prompts; per-prompt kwargs (e.g.
        per-prompt ``attention_mask``) are not currently supported.

    Returns
    -------
    BatchCaptureResult
        Contains per-prompt :class:`CaptureResult` objects and
        optionally ``aligned_attention`` of shape
        ``(n_prompts, n_layers, n_heads, n_positions, max_seq_len)``.

        ``aligned_attention`` is only populated when ``no_grad=True``
        (the default).  In gradient mode (``no_grad=False``), attention
        lives in each result's ``attention_tensors`` field and
        alignment is skipped because numpy padding would break the
        computation graph.

        Padded positions in ``aligned_attention`` are zero-filled and
        are **not** valid simplices.  Use ``metadata["seq_lens"]`` to
        identify each prompt's valid region before passing to simplex
        geometry functions.
    """
    if not prompts:
        return BatchCaptureResult(results=[], metadata={"n_prompts": 0})

    results = []
    for i, prompt in enumerate(prompts):
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)
        if prompt.ndim == 2 and prompt.shape[0] != 1:
            raise ValueError(
                f"Each prompt must be a single sequence. "
                f"Got batch size {prompt.shape[0]} at index "
                f"{i}. Use separate entries in the prompts "
                f"list for multiple sequences."
            )
        r = capture_forward(
            model, prompt,
            predictions=predictions,
            attention=attention,
            hidden_states=hidden_states,
            raw_hidden_states=raw_hidden_states,
            routing=routing,
            policy=policy,
            no_grad=no_grad,
            **model_kwargs,
        )
        results.append(r)

    seq_lens = [r.metadata.get("seq_len", 0) for r in results]

    aligned = None
    if (
        attention
        and no_grad
        and all(r.attention is not None for r in results)
    ):
        aligned = _align_attention(
            results, policy or SamplingPolicy()
        )

    return BatchCaptureResult(
        results=results,
        aligned_attention=aligned,
        metadata={
            "n_prompts": len(prompts),
            "seq_lens": seq_lens,
            "max_seq_len": max(seq_lens) if seq_lens else 0,
        },
    )


def _align_attention(
    results: list[CaptureResult],
    policy: SamplingPolicy,
) -> np.ndarray:
    """Pad and stack attention arrays to a common shape.

    Returns shape ``(n_prompts, n_layers, n_heads, n_positions, max_seq_len)``.

    .. warning::

       Padded positions are zero-filled and are **not** valid simplices.
       Use ``metadata["seq_lens"]`` to identify the valid region for
       each prompt before passing to simplex geometry functions.
    """
    from fisher_torch.convert import stack_attention

    n_heads_cfg = results[0].metadata.get("n_heads", 1)
    n_heads_sel = len(policy.selected_heads(n_heads_cfg))

    stacked_per_prompt = []
    for r in results:
        # stack_attention -> (n_layers, n_heads, seq_len) or
        #                    (n_layers, n_heads, n_positions, seq_len)
        s = stack_attention(r.attention, n_heads=n_heads_sel)
        if s.ndim == 3:
            # Single position: add n_positions axis.
            s = s[:, :, np.newaxis, :]
        stacked_per_prompt.append(s)

    max_pos = max(s.shape[2] for s in stacked_per_prompt)
    max_seq = max(s.shape[3] for s in stacked_per_prompt)

    padded = []
    for s in stacked_per_prompt:
        pos_pad = max_pos - s.shape[2]
        seq_pad = max_seq - s.shape[3]
        if pos_pad > 0 or seq_pad > 0:
            pad_width = [
                (0, 0),  # n_layers
                (0, 0),  # n_heads
                (0, pos_pad),  # n_positions
                (0, seq_pad),  # seq_len
            ]
            s = np.pad(s, pad_width, mode="constant",
                        constant_values=0.0)
        padded.append(s)

    return np.stack(padded, axis=0)


def extract_for_atlas(
    model,
    tokenizer,
    prompts: list[str],
    *,
    layers: list[int] | None = None,
    **model_kwargs,
) -> tuple[np.ndarray, list[int]]:
    """Extract atlas-position attention simplices for multiple prompts.

    Convenience wrapper around :func:`capture_batch` that tokenizes
    prompts, uses the ``"atlas"`` position preset, and returns a dense
    float64 array ready for geometric analysis.

    Parameters
    ----------
    model
        A HuggingFace ``PreTrainedModel`` (or compatible mock).
    tokenizer
        A HuggingFace tokenizer (or any object whose ``.encode(text,
        return_tensors="pt")`` returns a ``(1, seq_len)`` tensor).
    prompts : list[str]
        Text prompts to process.
    layers : list[int] or None, optional
        Which transformer layers to extract.  ``None`` extracts all.
    **model_kwargs
        Extra kwargs forwarded to each model forward call
        (e.g. ``attention_mask``).

    Returns
    -------
    attention : np.ndarray
        Float64 array of shape
        ``(n_prompts, n_positions, n_layers, n_heads, max_seq_len)``.
        Padded positions are zero-filled and are **not** valid
        simplices — use *seq_lens* to identify valid regions.
    seq_lens : list[int]
        Token count for each prompt.
    """
    if not prompts:
        return np.empty(0, dtype=np.float64), []

    token_ids = [tokenizer.encode(p, return_tensors="pt") for p in prompts]

    policy = SamplingPolicy(
        position_preset="atlas",
        layers=layers,
    )

    batch = capture_batch(
        model,
        token_ids,
        predictions=False,
        attention=True,
        policy=policy,
        **model_kwargs,
    )

    seq_lens = batch.metadata["seq_lens"]

    # aligned_attention shape: (n_prompts, n_layers, n_heads, n_positions, max_seq_len)
    # Transpose to:            (n_prompts, n_positions, n_layers, n_heads, max_seq_len)
    aligned = batch.aligned_attention
    attention = np.transpose(aligned, (0, 3, 1, 2, 4)).astype(np.float64)

    return attention, seq_lens

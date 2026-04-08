# Changelog

## 0.3.1

### Docs

- README and CHANGELOG updated for v0.3.0 features (missed in initial
  0.3.0 build uploaded to PyPI).

## 0.3.0

### New features

- **`extract_for_atlas` convenience function.**  Single call: model +
  tokenizer + prompts + target layers in, dense float64 array of shape
  `(n_prompts, n_positions, n_layers, n_heads, max_seq_len)` + seq_lens
  out.  Replaces the duplicated tokenize-batch-transpose loop across
  replay scripts.

### Tests

- Atlas position preset edge-case tests at seq_len 1, 2, 5, 8, 128,
  512 — locks down the `early/mid/late/final` formula and deduplication
  behavior.
- Attention float64 guarantee test on the numpy path.

### Docs

- `CaptureResult` and `capture_forward` docstrings now document the
  float64 guarantee: all numpy arrays on the `no_grad=True` path are
  float64, enforced by `to_simplex_array`.

## 0.2.0

### Breaking changes

- `to_simplex_array` now uses `renormalize="always"` instead of
  `renormalize="warn"`.  Float32-to-float64 drift is silently corrected
  rather than emitting a `UserWarning`.  Callers that relied on the
  warning to detect drift should use `fisher_simplex.validate_simplex`
  directly with `renormalize="warn"`.
- `extract_attention` and `extract_layerwise_predictions` return
  `list[dict]` / `list[list[dict]]` for batch > 1 instead of raising
  `ValueError`.
- `truncate_and_renormalize` moved from `utils` to `convert`.
  A re-export in `utils` preserves backward compatibility.

### New features

- **Gradient-enabled extraction** (`no_grad=False`).  All extractors
  accept `return_tensors=True`; `CaptureResult` gains `*_tensors`
  fields and `detach_to_numpy()`.  `safe_softmax` gains
  `differentiable=True` to preserve clean gradient flow.
- **Raw hidden states** via `extract_hidden_states()` and
  `capture_forward(..., raw_hidden_states=True)`.  Returns hidden
  state vectors without lm_head projection.
- **Multi-prompt batch** via `capture_batch()` and
  `BatchCaptureResult`.  Variable-length prompts with attention
  alignment (zero-padded to common shape).
- **Position presets** (`position_preset="atlas"` / `"quartiles"`)
  on `SamplingPolicy`, with `position_labels()` for named positions.
- **GQA group tracking** in `CaptureResult.gqa_groups`, aligned to
  `policy.selected_heads()`.  Non-uniform GQA layouts produce `None`.

### Bug fixes

- `capture_forward` now moves all tensor kwargs (e.g. `attention_mask`)
  to the model's input device, not just `input_ids`.
- `_align_attention` pads both `n_positions` and `seq_len` axes,
  fixing `ValueError` when position presets produce different counts
  across prompts.
- `_extract_attention_single` returns empty arrays instead of crashing
  when heads or positions selection is empty.
- `capture_batch` validates that each prompt is a single sequence
  (batch dimension = 1).

## 0.1.0

Initial release.

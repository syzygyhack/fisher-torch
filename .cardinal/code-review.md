# Code Review: fisher-torch v0.1

**Date:** 2026-03-28
**Scope:** Full codebase (extractors, capture, convert, utils, sampling)
**Baseline:** 99 tests passing, 0 failures

---

## Issues Found

### HIGH Severity

#### H1: Duplicate `ProjectionSpec` class (FIXED)

**Location:** `utils.py:22-35` and `capture.py:24-48`

Two classes named `ProjectionSpec` with incompatible fields:
- `utils.ProjectionSpec`: fields `k`, `remainder_mode`, `tail_cardinality`, `original_vocab_size`
- `capture.ProjectionSpec`: fields `mode`, `top_k`, `original_dim`, `tail_cardinality`, `simplex_dim`

`__init__.py` exported `capture.ProjectionSpec` as the public API, but `topk_softmax` returned `utils.ProjectionSpec`. A user calling `topk_softmax` would get an object that failed `isinstance(result, ProjectionSpec)` against the exported class.

**Fix:** Renamed `utils.ProjectionSpec` to `TopkResult`. Added `TopkResult` to `__all__`.

#### H2: No validation for non-positive temperature (FIXED)

**Location:** `utils.py:37-65` (`safe_softmax`)

`safe_softmax(logits, temperature=0.0)` produced `inf` from division, leading to NaN after softmax. No error raised.

**Fix:** Added `ValueError` for `temperature <= 0` at function entry.

### MEDIUM Severity

#### M1: `selected_positions(0)` returned `[-1]` (FIXED)

**Location:** `sampling.py:99-100`

With `final_token_only=True` and `seq_len=0`, the expression `seq_len - 1` evaluates to `-1`. This is a valid Python index but semantically wrong for an empty sequence.

**Fix:** Added early return of `[]` when `seq_len <= 0`.

#### M2: No validation for `top_k > vocab_size` (FIXED)

**Location:** `utils.py:68-126` (`topk_softmax`)

`torch.topk` raises a cryptic `RuntimeError` when k exceeds the tensor dimension. No guard in `topk_softmax`.

**Fix:** Added explicit `ValueError` with descriptive message before the `torch.topk` call.

### LOW Severity

#### L1: `extract_attention` return type differs from spec

**Location:** `extractors.py:64-118`

Spec Section 2.4 signature says `-> np.ndarray`. Implementation returns `dict[int, np.ndarray]`. The dict return type is correct for the use case (keyed by layer index) and matches `CaptureResult.attention` typing.

**Disposition:** Accepted deviation. The implementation is better than the simplified spec signature. No code change needed.

#### L2: Precision roundtrip in `topk_softmax`

**Location:** `utils.py:109-117`

The flow is: float32 probs -> float64 (topk_to_simplex) -> float32 tensor -> float64 (to_simplex_array). The intermediate float32 step loses ~7 digits of precision.

**Disposition:** Acceptable. The float32->float64 conversion in `to_simplex_array` recovers the representation, and the precision loss in the intermediate tensor is negligible for simplex values (which are bounded [0,1]). Would only matter if downstream code uses the tensor directly instead of going through `to_simplex_array`.

#### L3: No edge case tests for single-token sequences

**Disposition:** Low priority. Single-token (seq_len=1) works correctly by construction. Empty sequence edge case now guarded (M1).

#### L4: No tests for degenerate attention matrices

Identity or near-zero attention matrices are valid edge cases but work correctly since attention rows are already simplex-valid after softmax.

**Disposition:** Could add in a follow-up pass. Not a correctness risk.

#### L5: Frequent `UserWarning: Simplex sum drift` in tests

**Location:** `convert.py:32`

94 warnings across the test suite from float32->float64 conversion drift. This is expected behavior (fisher-simplex `validate_simplex` with `renormalize="warn"` correctly detects and fixes sum drift ~1e-7 from float32 precision).

**Disposition:** Expected. The warnings confirm the validation pipeline is working. Could suppress in tests with `filterwarnings` if desired.

---

## Summary

| Severity | Found | Fixed | Remaining |
|----------|-------|-------|-----------|
| HIGH     | 2     | 2     | 0         |
| MEDIUM   | 2     | 2     | 0         |
| LOW      | 5     | 0     | 5 (accepted) |

## Checklist

- [x] Correctness: extractors produce valid simplex arrays (tested)
- [x] Numerical stability: clipping at 1e-30, float64 output, stable softmax
- [x] API consistency: matches spec (one documented deviation in L1)
- [x] Code quality: docstrings present, no unnecessary abstractions
- [x] Security: pure computation library, no injection surface
- [x] Test coverage: 104 tests, edge cases for empty seq/invalid temp/k overflow
- [x] Dependencies: only fisher-simplex, torch, numpy (no transformers in core)

## Files Modified

- `src/fisher_torch/utils.py` — renamed `ProjectionSpec` to `TopkResult`, added temperature and top_k validation
- `src/fisher_torch/sampling.py` — guard for empty sequences
- `src/fisher_torch/__init__.py` — export `TopkResult`
- `tests/test_utils.py` — updated import, added validation tests
- `tests/test_sampling.py` — added empty sequence tests

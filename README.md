# fisher-torch

Extract simplex arrays from PyTorch model outputs.

Built on [fisher-simplex](https://github.com/syzygyhack/fisher-simplex) for simplex geometry and invariants.

## Install

```bash
pip install fisher-torch
```

With transformers support:

```bash
pip install fisher-torch[transformers]
```

## Quick start

```python
import torch
from fisher_torch import extract_predictions, capture_forward
from fisher_torch import SamplingPolicy

# From raw logits
logits = torch.randn(10, 50000)  # (seq_len, vocab_size)
simplex = extract_predictions(logits, top_k=50)
# simplex.shape == (10, 51)  — top-50 + remainder bin, valid simplex

# From a HuggingFace model
policy = SamplingPolicy(top_k=50, final_token_only=True)
result = capture_forward(model, input_ids, predictions=True, policy=policy)
# result.predictions is a float64 numpy simplex array
# result.projection_spec describes the projection geometry
```

### Atlas extraction (convenience)

```python
from fisher_torch import extract_for_atlas

attention, seq_lens = extract_for_atlas(model, tokenizer, texts, layers=[0, 15, 31])
# attention.shape == (n_prompts, n_positions, n_layers, n_heads, max_seq_len)
# attention.dtype == np.float64
# seq_lens gives each prompt's valid token count
```

### Multi-prompt batch extraction

```python
from fisher_torch import capture_batch, SamplingPolicy

prompts = [tokenizer.encode(p, return_tensors="pt") for p in texts]
policy = SamplingPolicy(position_preset="atlas")
batch = capture_batch(model, prompts, attention=True, policy=policy)
# batch.aligned_attention.shape ==
#   (n_prompts, n_layers, n_heads, n_positions, max_seq_len)
# batch.metadata["seq_lens"] gives each prompt's valid length
```

### Gradient-enabled extraction

```python
result = capture_forward(
    model, input_ids,
    predictions=True, attention=True,
    no_grad=False,
)
# result.prediction_tensors — torch.Tensor with grad graph intact
# result.attention_tensors  — dict[int, Tensor]
result.detach_to_numpy()  # convert tensor fields to numpy in-place
```

### Raw hidden states

```python
result = capture_forward(
    model, input_ids,
    predictions=False, hidden_states=True, raw_hidden_states=True,
)
# result.hidden_states     — logit-lens projected predictions per layer
# result.raw_hidden_states — raw vectors without lm_head projection
```

## Modules

| Module | Purpose |
|--------|---------|
| `extractors` | Stateless functions: logits, attention, hidden states, routing → simplex arrays |
| `capture` | `capture_forward` (single pass), `capture_batch` (multi-prompt with alignment), `extract_for_atlas` (convenience) |
| `convert` | Tensor ↔ numpy simplex conversion, `stack_attention`, `truncate_and_renormalize` |
| `sampling` | `SamplingPolicy` — layers, heads, positions, presets (`"atlas"`, `"quartiles"`) |
| `utils` | Numerically stable softmax, top-k simplex projection, device helpers |

# fisher-torch

Extract simplex arrays from PyTorch model outputs.

Part of the [Fisher stack](https://github.com/syzygy/fisher-simplex) — built on
[fisher-simplex](https://github.com/syzygy/fisher-simplex) for simplex geometry
and invariants.

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

## Modules

| Module | Purpose |
|--------|---------|
| `extractors` | Stateless functions: logits, attention, hidden states, routing → simplex arrays |
| `capture` | Single forward pass orchestrator (`capture_forward`) |
| `convert` | Tensor ↔ numpy simplex array conversion |
| `sampling` | `SamplingPolicy` — controls which layers, heads, positions to extract |
| `utils` | Numerically stable softmax, top-k simplex projection |

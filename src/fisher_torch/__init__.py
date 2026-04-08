"""Fisher-torch: extract simplex arrays from PyTorch model outputs."""

from fisher_torch.capture import (
    BatchCaptureResult,
    CaptureResult,
    ProjectionSpec,
    capture_batch,
    capture_forward,
)
from fisher_torch.convert import (
    from_simplex_array,
    stack_attention,
    to_simplex_array,
    truncate_and_renormalize,
)
from fisher_torch.extractors import (
    extract_attention,
    extract_hidden_states,
    extract_layerwise_predictions,
    extract_predictions,
    extract_routing,
)
from fisher_torch.sampling import SamplingPolicy
from fisher_torch.utils import (
    TopkResult,
    get_input_device,
    safe_softmax,
    topk_softmax,
)

__version__ = "0.2.0"

__all__ = [
    # convert
    "to_simplex_array",
    "from_simplex_array",
    "stack_attention",
    "truncate_and_renormalize",
    # utils
    "safe_softmax",
    "topk_softmax",
    "TopkResult",
    "get_input_device",
    # sampling
    "SamplingPolicy",
    # extractors
    "extract_predictions",
    "extract_attention",
    "extract_hidden_states",
    "extract_layerwise_predictions",
    "extract_routing",
    # capture
    "capture_forward",
    "capture_batch",
    "CaptureResult",
    "BatchCaptureResult",
    "ProjectionSpec",
]

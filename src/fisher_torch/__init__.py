"""Fisher-torch: extract simplex arrays from PyTorch model outputs."""

from fisher_torch.capture import CaptureResult, ProjectionSpec, capture_forward
from fisher_torch.convert import from_simplex_array, stack_attention, to_simplex_array
from fisher_torch.extractors import (
    extract_attention,
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
    truncate_and_renormalize,
)

__version__ = "0.1.0"

__all__ = [
    # convert
    "to_simplex_array",
    "from_simplex_array",
    "stack_attention",
    # utils
    "safe_softmax",
    "topk_softmax",
    "TopkResult",
    "get_input_device",
    "truncate_and_renormalize",
    # sampling
    "SamplingPolicy",
    # extractors
    "extract_predictions",
    "extract_attention",
    "extract_layerwise_predictions",
    "extract_routing",
    # capture
    "capture_forward",
    "CaptureResult",
    "ProjectionSpec",
]

"""Fisher-torch: extract simplex arrays from PyTorch model outputs."""

from fisher_torch.convert import from_simplex_array, to_simplex_array
from fisher_torch.sampling import SamplingPolicy
from fisher_torch.utils import ProjectionSpec, safe_softmax, topk_softmax

__version__ = "0.1.0"

__all__ = [
    "from_simplex_array",
    "to_simplex_array",
    "safe_softmax",
    "topk_softmax",
    "ProjectionSpec",
    "SamplingPolicy",
]

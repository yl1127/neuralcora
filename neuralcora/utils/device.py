"""Device resolution helpers for NeuralCora."""
from __future__ import annotations

from typing import Tuple

import torch

_PREFERRED_DEVICES: Tuple[str, ...] = ("cuda", "mps", "cpu")


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a torch.device from a user preference string.

    Parameters
    ----------
    preference: str
        Desired device name. ``"auto"`` (default) chooses the first available
        backend from CUDA ➜ MPS ➜ CPU.
    """

    pref = (preference or "").lower()
    if pref == "auto" or pref == "" or pref not in _PREFERRED_DEVICES:
        candidates = _PREFERRED_DEVICES
    else:
        candidates = (pref,)

    for candidate in candidates:
        if candidate == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if candidate == "mps":
            mps_available = getattr(torch.backends, "mps", None)
            if mps_available and torch.backends.mps.is_available():
                return torch.device("mps")
        if candidate == "cpu":
            return torch.device("cpu")

    # Fallback to CPU if nothing matched
    return torch.device("cpu")

"""Reproducibility + environment introspection helpers.

Deliberately import-light at module load so this works on macOS/CPU during
structural validation. Heavy libs (torch) are imported lazily inside functions.
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class GPUInfo:
    cuda_available: bool
    device_name: str | None
    total_vram_gb: float | None
    compute_capability: str | None
    torch_version: str | None
    supports_bf16: bool
    vllm_feasible: bool  # compute capability >= 8.0 (Ampere+)


def get_gpu_info() -> GPUInfo:
    """Probe the active accelerator. Safe to call on CPU-only machines."""
    try:
        import torch
    except ImportError:
        return GPUInfo(False, None, None, None, None, False, False)

    if not torch.cuda.is_available():
        return GPUInfo(False, None, None, None, torch.__version__, False, False)

    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    cap_str = f"{cap[0]}.{cap[1]}"
    # bf16 + vLLM both effectively require Ampere (compute capability 8.0+).
    # The T4 is Turing (7.5): no native bf16, and many vLLM builds reject it.
    supports_bf16 = cap[0] >= 8
    return GPUInfo(
        cuda_available=True,
        device_name=props.name,
        total_vram_gb=round(props.total_memory / 1024**3, 2),
        compute_capability=cap_str,
        torch_version=torch.__version__,
        supports_bf16=supports_bf16,
        vllm_feasible=cap[0] >= 8,
    )


def env_fingerprint() -> dict:
    """A JSON-serializable snapshot for logging alongside every run."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": asdict(get_gpu_info()),
    }
    for pkg in ("transformers", "peft", "trl", "datasets", "lm_eval", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[pkg] = "not-installed"
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        info["git_commit"] = "unknown"
    return info


def print_env() -> None:
    print(json.dumps(env_fingerprint(), indent=2))


if __name__ == "__main__":
    print_env()

#!/usr/bin/env python3
"""Validate the official OpenPI environment without downloading weights or opening hardware."""

from __future__ import annotations

import json
import sys

import jax
import torch


def main() -> int:
    import openpi

    vram_gib = None
    gpu_name = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        vram_gib = round(properties.total_memory / 2**30, 2)
        gpu_name = torch.cuda.get_device_name(0)

    payload = {
        "event": "nero_pi05_environment_verified",
        "python": sys.version.split()[0],
        "openpi_path": openpi.__file__,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu": gpu_name,
        "gpu_memory_gib": vram_gib,
        "pi05_local_weight_loading_approved": bool(vram_gib is not None and vram_gib > 8.0),
    }
    print(json.dumps(payload, indent=2))
    if not payload["torch_cuda_available"]:
        raise SystemExit("CUDA is unavailable to OpenPI; do not load pi0.5 weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

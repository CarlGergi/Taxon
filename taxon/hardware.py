"""What a result was measured on. Every result file records it."""

from __future__ import annotations

import os
import platform
import subprocess
from functools import lru_cache


def accelerator() -> str | None:
    """The device that trains the real model here: an NVIDIA GPU, else Apple's GPU, else none."""
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _linux_value(path: str, key: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


@lru_cache(maxsize=1)
def _machine() -> dict:
    if platform.system() == "Darwin":
        chip = _sysctl("machdep.cpu.brand_string")
        memory = _sysctl("hw.memsize")
        memory_gb = round(int(memory) / 2**30) if memory else None
    else:
        chip = _linux_value("/proc/cpuinfo", "model name")
        kib = _linux_value("/proc/meminfo", "MemTotal")
        memory_gb = round(int(kib.split()[0]) / 2**20) if kib else None
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "chip": chip,
        "cpu_cores": os.cpu_count(),
        "memory_gb": memory_gb,
    }


def hardware(device: str = "cpu") -> dict:
    """The machine, plus the device that did the work and its name."""
    record = {**_machine(), "device": device}
    if device == "cuda":
        import torch

        record["gpu"] = torch.cuda.get_device_name(0)
    elif device == "mps":
        record["gpu"] = f"{record['chip']} GPU"
    return record

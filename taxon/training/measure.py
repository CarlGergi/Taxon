"""Measure memory and step time for real fine-tuning and GRPO on this machine, to size runs.

This runs the same trainers as a real run, for a few steps, on provisional labels. It saves no
model anyone should use and computes no accuracy: only time and memory, with the hardware.
"""

from __future__ import annotations

import math
import os
import resource
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import median

import torch
from transformers import TrainerCallback

from taxon.hardware import hardware
from taxon.pins import MODEL, MODEL_REVISION

OVERNIGHT_HOURS = 8.0
_GB = 2**30


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _allocated(device: str) -> tuple[float | None, float | None]:
    """Memory the framework holds for tensors, and what the GPU driver has reserved, in GB.

    On MPS only the driver's figure counts the weights: loaded from safetensors, they never pass
    through the allocator, which reads near zero with the whole model on the GPU.
    """
    if device == "mps":
        return None, torch.mps.driver_allocated_memory() / _GB
    if device == "cuda":
        return torch.cuda.memory_allocated() / _GB, torch.cuda.memory_reserved() / _GB
    return None, None


def max_rss_gb() -> float:
    """Peak resident memory of this process. macOS reports bytes, Linux kilobytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / _GB if peak > 2**32 else peak * 1024 / _GB


class StepMeter(TrainerCallback):
    """Wall time of each optimizer step, and memory sampled at its end."""

    def __init__(self, device: str):
        self.device = device
        self.seconds: list[float] = []
        self.allocated: list[float] = []
        self.driver: list[float] = []
        self._start = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        _synchronize(self.device)
        self._start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        _synchronize(self.device)
        self.seconds.append(time.perf_counter() - self._start)
        allocated, driver = _allocated(self.device)
        if allocated is not None:
            self.allocated.append(allocated)
        if driver is not None:
            self.driver.append(driver)

    def summary(self) -> dict:
        """The first step warms caches and compiles kernels, so it is reported but not averaged."""
        steady = self.seconds[1:] or self.seconds
        return {
            "steps": len(self.seconds),
            "first_step_s": self.seconds[0] if self.seconds else None,
            "median_step_s": median(steady) if steady else None,
            "peak_allocated_gb": max(self.allocated) if self.allocated else None,
            "peak_driver_gb": max(self.driver) if self.driver else None,
        }


def overnight_steps(median_step_s: float, hours: float = OVERNIGHT_HOURS) -> int:
    """Steps that fit in `hours`, leaving a tenth for loading, saving and slow steps."""
    return math.floor(hours * 3600 * 0.9 / median_step_s)


def measurement_record(
    device: str, phases: dict, settings: dict, rows: Sequence[dict], hours: float
) -> dict:
    record = {
        "hardware": hardware(device),
        "model": f"{MODEL}@{MODEL_REVISION}",
        "settings": settings,
        "examples": len(rows),
        "with_image": sum(bool(row["image_path"]) for row in rows),
        "process_peak_rss_gb": max_rss_gb(),
        # Other work on the machine slows steps, so the load at the end is kept with the timings.
        "load_average_1_5_15": list(os.getloadavg()),
        **phases,
        "overnight_hours": hours,
    }
    for phase in phases.values():
        if phase.get("median_step_s"):
            phase["steps_overnight"] = overnight_steps(phase["median_step_s"], hours)
    return record


def write_measurement(path: Path, record: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

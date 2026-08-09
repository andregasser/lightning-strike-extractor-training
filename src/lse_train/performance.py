from __future__ import annotations

import sys
from typing import Any


def _device_type(device: Any) -> str:
    return str(getattr(device, "type", device)).split(":", 1)[0]


class DeviceProfiler:
    """Collect only device metrics exposed reliably by the active PyTorch backend."""

    def __init__(self, torch: Any, device: Any) -> None:
        self.torch = torch
        self.device = device
        self.device_type = _device_type(device)
        self.peak_mps_allocated_bytes: int | None = None
        self.peak_mps_driver_bytes: int | None = None

    def start(self) -> None:
        self.synchronize()
        if self.device_type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        self.sample_memory()

    def synchronize(self) -> None:
        if self.device_type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device_type == "mps" and hasattr(self.torch.mps, "synchronize"):
            self.torch.mps.synchronize()

    def sample_memory(self) -> None:
        if self.device_type != "mps":
            return
        if hasattr(self.torch.mps, "current_allocated_memory"):
            current = int(self.torch.mps.current_allocated_memory())
            self.peak_mps_allocated_bytes = max(self.peak_mps_allocated_bytes or 0, current)
        if hasattr(self.torch.mps, "driver_allocated_memory"):
            current = int(self.torch.mps.driver_allocated_memory())
            self.peak_mps_driver_bytes = max(self.peak_mps_driver_bytes or 0, current)

    def report(self) -> dict[str, Any]:
        if self.device_type == "cuda":
            memory = {
                "measurement_status": "available",
                "method": "torch.cuda peak memory statistics",
                "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved(self.device)),
            }
        elif self.device_type == "mps" and self.peak_mps_allocated_bytes is not None:
            memory = {
                "measurement_status": "sampled",
                "method": "PyTorch MPS memory sampled at synchronized phase boundaries",
                "peak_sampled_allocated_bytes": self.peak_mps_allocated_bytes,
                "peak_sampled_driver_allocated_bytes": self.peak_mps_driver_bytes,
                "limitation": "PyTorch MPS does not expose resettable peak memory statistics.",
            }
        else:
            memory = {
                "measurement_status": "unavailable",
                "method": None,
                "reason": f"PyTorch exposes no device peak-memory metric for {self.device_type}.",
            }
        return {
            "type": self.device_type,
            "memory": memory,
            "utilization": {
                "measurement_status": "unavailable",
                "average_percent": None,
                "reason": (
                    f"Reliable {self.device_type} utilization sampling is not exposed by the "
                    "configured dependency-free PyTorch measurement path."
                ),
            },
        }


def process_peak_memory_report(
    *, max_rss: float | None = None, platform: str | None = None
) -> dict[str, Any]:
    """Return process-lifetime peak RSS with platform-correct units."""
    platform = platform or sys.platform
    if max_rss is None:
        try:
            import resource
        except ImportError:
            return {
                "measurement_status": "unavailable",
                "peak_resident_set_bytes": None,
                "reason": "The Python resource module is unavailable on this platform.",
            }
        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    multiplier = 1 if platform == "darwin" else 1024
    return {
        "measurement_status": "available",
        "peak_resident_set_bytes": int(max_rss * multiplier),
        "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "scope": "process lifetime; the counter cannot be reset at training start",
    }


def build_performance_report(
    *,
    total_seconds: float,
    training_seconds: float,
    validation_seconds: float,
    data_loading_seconds: float,
    device_transfer_seconds: float,
    optimization_seconds: float,
    training_images: int,
    training_batches: int,
    validation_images: int,
    validation_batches: int,
    epochs: list[dict[str, Any]],
    process_memory: dict[str, Any],
    device: dict[str, Any],
) -> dict[str, Any]:
    accounted = data_loading_seconds + device_transfer_seconds + optimization_seconds
    return {
        "measurement_method": {
            "clock": "time.perf_counter",
            "device_synchronization": "before phase boundaries on CUDA and MPS",
            "data_loading": "time waiting for the next DataLoader batch",
            "throughput_scope": "training phases include data loading, transfer, and optimization",
        },
        "total_seconds": total_seconds,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "unattributed_total_overhead_seconds": max(
            0.0, total_seconds - training_seconds - validation_seconds
        ),
        "data_loading_seconds": data_loading_seconds,
        "device_transfer_seconds": device_transfer_seconds,
        "optimization_seconds": optimization_seconds,
        "unattributed_training_overhead_seconds": max(0.0, training_seconds - accounted),
        "data_loading_fraction": (
            data_loading_seconds / training_seconds if training_seconds else None
        ),
        "training_images": training_images,
        "training_batches": training_batches,
        "validation_images": validation_images,
        "validation_batches": validation_batches,
        "training_images_per_second": (
            training_images / training_seconds if training_seconds else None
        ),
        "training_batches_per_second": (
            training_batches / training_seconds if training_seconds else None
        ),
        "average_training_batch_seconds": (
            training_seconds / training_batches if training_batches else None
        ),
        "epochs": epochs,
        "process_memory": process_memory,
        "device": device,
    }

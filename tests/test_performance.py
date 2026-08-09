from __future__ import annotations

from types import SimpleNamespace

from lse_train.performance import (
    DeviceProfiler,
    build_performance_report,
    process_peak_memory_report,
)


def test_converts_process_peak_memory_units_by_platform() -> None:
    assert process_peak_memory_report(max_rss=2048, platform="darwin")[
        "peak_resident_set_bytes"
    ] == 2048
    assert process_peak_memory_report(max_rss=2048, platform="linux")[
        "peak_resident_set_bytes"
    ] == 2 * 1024 * 1024


def test_reports_cuda_peak_memory_and_unavailable_utilization() -> None:
    calls: list[tuple[str, object]] = []
    cuda = SimpleNamespace(
        synchronize=lambda device: calls.append(("synchronize", device)),
        reset_peak_memory_stats=lambda device: calls.append(("reset", device)),
        max_memory_allocated=lambda device: 123,
        max_memory_reserved=lambda device: 456,
    )
    profiler = DeviceProfiler(SimpleNamespace(cuda=cuda), "cuda:0")

    profiler.start()
    report = profiler.report()

    assert calls == [("synchronize", "cuda:0"), ("reset", "cuda:0")]
    assert report["memory"] == {
        "measurement_status": "available",
        "method": "torch.cuda peak memory statistics",
        "peak_allocated_bytes": 123,
        "peak_reserved_bytes": 456,
    }
    assert report["utilization"]["average_percent"] is None


def test_reports_sampled_mps_memory() -> None:
    allocated = iter((100, 150))
    driver = iter((200, 180))
    mps = SimpleNamespace(
        synchronize=lambda: None,
        current_allocated_memory=lambda: next(allocated),
        driver_allocated_memory=lambda: next(driver),
    )
    profiler = DeviceProfiler(SimpleNamespace(mps=mps), "mps")

    profiler.start()
    profiler.sample_memory()
    report = profiler.report()

    assert report["memory"]["measurement_status"] == "sampled"
    assert report["memory"]["peak_sampled_allocated_bytes"] == 150
    assert report["memory"]["peak_sampled_driver_allocated_bytes"] == 200


def test_reports_cpu_device_memory_as_unavailable() -> None:
    report = DeviceProfiler(SimpleNamespace(), "cpu").report()

    assert report["type"] == "cpu"
    assert report["memory"]["measurement_status"] == "unavailable"
    assert report["utilization"]["average_percent"] is None


def test_builds_throughput_and_phase_report() -> None:
    report = build_performance_report(
        total_seconds=15.0,
        training_seconds=10.0,
        validation_seconds=4.0,
        data_loading_seconds=2.0,
        device_transfer_seconds=1.0,
        optimization_seconds=6.0,
        training_images=20,
        training_batches=10,
        validation_images=4,
        validation_batches=2,
        epochs=[{"epoch": 1, "total_seconds": 15.0}],
        process_memory={"peak_resident_set_bytes": 1000},
        device={"type": "cpu"},
    )

    assert report["training_images_per_second"] == 2.0
    assert report["training_batches_per_second"] == 1.0
    assert report["average_training_batch_seconds"] == 1.0
    assert report["data_loading_fraction"] == 0.2
    assert report["unattributed_training_overhead_seconds"] == 1.0
    assert report["unattributed_total_overhead_seconds"] == 1.0

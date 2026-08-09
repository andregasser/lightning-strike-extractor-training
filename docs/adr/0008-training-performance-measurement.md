# ADR 0008: Record conservative training performance measurements

## Status

Accepted

## Context

Potential optimizations such as DataLoader workers, prefetching, larger batches,
mixed precision and backend changes should be selected from measured
bottlenecks. The training report previously contained losses and optimization
configuration but no evidence about throughput, phase duration or memory use.

GPU work is asynchronous, platform memory APIs have different semantics and
PyTorch does not expose one reliable utilization percentage across CPU, CUDA
and Apple MPS. Reporting unsynchronized durations or guessed utilization would
look precise while producing misleading comparisons.

## Decision

Schema-version-three training reports contain a `performance` section with:

- total and per-epoch wall-clock duration measured by `time.perf_counter`;
- DataLoader wait, device transfer, optimization and validation duration;
- processed image and batch counts and training-phase throughput;
- process peak resident memory from `resource.getrusage`;
- backend-specific memory and utilization availability metadata.

CUDA and MPS are synchronized at phase boundaries. CUDA memory uses resettable
`torch.cuda` peak allocated and reserved counters. PyTorch MPS has no equivalent
peak counter, so current allocated and driver-allocated memory are sampled after
synchronized training and validation phases and labeled `sampled`. Process RSS
is labeled as a process-lifetime peak because `ru_maxrss` cannot be reset.

No external monitoring dependency or system command is introduced. Device
utilization is represented by `average_percent: null`, an `unavailable` status
and a backend-specific reason until a reliable supported measurement path is
selected. Missing capability must never be represented as zero utilization.

## Consequences

- Training runs provide evidence for prioritizing data pipeline, memory and
  backend optimization work.
- Synchronization adds measurement overhead, especially on asynchronous GPU
  backends; comparisons remain meaningful when all candidate runs use the same
  instrumentation.
- CUDA memory values are true training-window peaks, MPS values are sampled
  upper observations and process RSS covers the whole process lifetime.
- Phase totals may not exactly equal total wall time because Python control,
  scheduler steps, checkpoint-state copies and report preparation are separate;
  unattributed training-phase and total-loop overhead are reported explicitly.
- Utilization optimization remains deferred until a dependable implementation
  can be added without pretending that incomparable platform signals match.

## Alternatives considered

- Use unsynchronized wall-clock timings. This measures asynchronous dispatch on
  CUDA and MPS rather than completed work.
- Invoke `nvidia-smi` or vendor utilities. This adds platform dependencies and
  does not solve CPU or MPS comparability.
- Add a process-monitoring dependency immediately. It can provide CPU sampling
  but does not create a uniform accelerator-utilization contract.
- Omit unavailable fields. Explicit status, `null` and reason values make the
  schema stable and prevent consumers from confusing missing data with zero.

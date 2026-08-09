# Training roadmap

This roadmap tracks model-training work that is not yet implemented. Completed
items must be removed or marked complete only after code, focused tests,
documentation and relevant validation are present.

## Priority 1: establish a trustworthy baseline

- [x] Use Faster R-CNN with ResNet-50 FPN V2 and explicit random,
  ImageNet-backbone and COCO-detector initialization modes. Keep COCO as the
  default and retain identical comparison runs for all three modes.
- [ ] Add Average Precision, mAP at IoU 0.5, COCO-style mAP across IoU 0.5–0.95
  and precision-recall curves.
- [ ] Report metrics by source, camera and recording condition where metadata is
  available.
- [ ] Add dataset composition reports for positive and negative images, box
  counts, source distributions, cameras, recording conditions and rare cases.
- [ ] Audit split groups so contiguous or near-duplicate recordings cannot cross
  train, validation and test boundaries.
- [ ] Add a learning-rate scheduler and record its complete configuration and
  learning-rate history in the training report.

## Priority 2: improve robustness and reproducibility

- [ ] Establish a hard-negative-mining loop that collects reviewed false
  positives such as cloud edges, lamps, reflections, rain and camera artifacts
  into new immutable dataset releases.
- [ ] Add experiment tracking for dataset release, Git commit, architecture,
  initialization weights, hyperparameters, seed, hardware, runtime, learning
  curves and final checkpoint.
- [ ] Add resumable checkpoints containing model, optimizer, scheduler, epoch and
  random-number-generator states.
- [ ] Select confidence thresholds on validation data for documented operating
  points such as high recall and low false-positive modes.
- [ ] Evaluate score calibration so reported confidence values have a measurable
  probabilistic interpretation.
- [ ] Repeat important comparisons with multiple seeds and report variation.

## Priority 3: extend data and training efficiency

- [ ] Evaluate box-safe scale, translation, zoom and controlled crop
  augmentation.
- [ ] Evaluate realistic JPEG compression and sensor-noise augmentation.
- [ ] Evaluate very small rotations only if they preserve thin channel geometry.
- [ ] Add automatic mixed precision and gradient scaling on supported GPUs.
- [ ] Tune DataLoader workers, prefetching, transfer behavior and batch size from
  measured bottlenecks.
- [ ] Record throughput, data-loading time, device utilization and peak memory.

## Deferred experiments

- [ ] Evaluate source-grouped cross-validation after the fixed test set and
  multi-seed baseline are established.
- [ ] Evaluate test-time augmentation only if error analysis shows a likely gain
  that justifies slower inference and additional ONNX/product complexity.

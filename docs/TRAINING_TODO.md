# Training roadmap

This roadmap tracks model-training work that is not yet implemented. Completed
items must be removed or marked complete only after code, focused tests,
documentation and relevant validation are present.

Effort estimates include implementation, focused tests, documentation and the
minimum validation needed to make the result trustworthy. Expected benefit is
relative to the current project state and should be revisited after the first
verified dataset and baseline results exist.

## Priority 1: establish a trustworthy baseline

- [x] Use Faster R-CNN with ResNet-50 FPN V2 and explicit random,
  ImageNet-backbone and COCO-detector initialization modes. Keep COCO as the
  default and retain identical comparison runs for all three modes.
  **Effort:** Medium. **Expected benefit:** Very high. Transfers general visual
  features and detector localization behavior while preserving controlled
  baselines for measuring the actual gain.
- [x] Add Average Precision, mAP at IoU 0.5, COCO-style mAP across IoU 0.5–0.95
  and precision-recall curves.
  **Effort:** Medium. **Expected benefit:** Very high. Replaces single-threshold
  conclusions with standard detection-quality measurements across thresholds.
- [ ] Report metrics by source, camera and recording condition where metadata is
  available.
  **Effort:** Medium. **Expected benefit:** High. Reveals failures hidden by an
  acceptable aggregate score and identifies weak source domains.
- [x] Add dataset composition reports for positive and negative images, box
  counts, source distributions, cameras, recording conditions and rare cases.
  **Effort:** Medium. **Expected benefit:** High. Makes imbalance and missing
  coverage visible before they distort training or evaluation.
- [x] Audit split groups so contiguous or near-duplicate recordings cannot cross
  train, validation and test boundaries.
  **Effort:** Medium. **Expected benefit:** Very high. Prevents data leakage and
  unrealistically optimistic validation or test results.
- [ ] Add a learning-rate scheduler and record its complete configuration and
  learning-rate history in the training report.
  **Effort:** Low. **Expected benefit:** High. Improves convergence with little
  complexity and reduces dependence on one fixed learning rate.

## Priority 2: improve robustness and reproducibility

- [ ] Establish a hard-negative-mining loop that collects reviewed false
  positives such as cloud edges, lamps, reflections, rain and camera artifacts
  into new immutable dataset releases.
  **Effort:** Medium to high and ongoing. **Expected benefit:** Very high.
  Directly teaches the detector about the false alarms observed in real use.
- [ ] Add experiment tracking for dataset release, Git commit, architecture,
  initialization weights, hyperparameters, seed, hardware, runtime, learning
  curves and final checkpoint.
  **Effort:** Medium. **Expected benefit:** High. Makes comparisons reproducible
  and prevents model-selection decisions from losing their provenance.
- [ ] Add resumable checkpoints containing model, optimizer, scheduler, epoch and
  random-number-generator states.
  **Effort:** Medium. **Expected benefit:** Medium. Protects long runs from
  interruption and avoids restarting expensive training.
- [ ] Select confidence thresholds on validation data for documented operating
  points such as high recall and low false-positive modes.
  **Effort:** Medium. **Expected benefit:** High. Aligns model behavior with
  product goals instead of relying on an arbitrary fixed threshold.
- [ ] Evaluate score calibration so reported confidence values have a measurable
  probabilistic interpretation.
  **Effort:** Medium. **Expected benefit:** Medium to high. Makes confidence
  values more interpretable and improves threshold decisions.
- [ ] Repeat important comparisons with multiple seeds and report variation.
  **Effort:** Medium in engineering and high in compute. **Expected benefit:**
  Medium to high. Distinguishes stable improvements from random variation.

## Priority 3: extend data and training efficiency

- [ ] Evaluate box-safe scale, translation, zoom and controlled crop
  augmentation.
  **Effort:** Medium. **Expected benefit:** Medium to high. Improves robustness
  to framing and distance while preserving valid boxes.
- [ ] Evaluate realistic JPEG compression and sensor-noise augmentation.
  **Effort:** Medium. **Expected benefit:** Medium to high. Represents camera and
  codec degradation likely to appear in real source videos.
- [ ] Evaluate very small rotations only if they preserve thin channel geometry.
  **Effort:** Medium. **Expected benefit:** Low to medium. May help with tilted
  cameras, but can damage thin channels and therefore needs careful validation.
- [ ] Add automatic mixed precision and gradient scaling on supported GPUs.
  **Effort:** Medium. **Expected benefit:** Medium to high. Can reduce memory use
  and training time without changing model capacity.
- [ ] Tune DataLoader workers, prefetching, transfer behavior and batch size from
  measured bottlenecks.
  **Effort:** Medium. **Expected benefit:** Medium. Improves throughput when data
  loading or device transfer is the measured bottleneck.
- [ ] Record throughput, data-loading time, device utilization and peak memory.
  **Effort:** Low to medium. **Expected benefit:** Medium. Provides the evidence
  needed to prioritize performance work instead of guessing.

## Deferred experiments

- [ ] Evaluate source-grouped cross-validation after the fixed test set and
  multi-seed baseline are established.
  **Effort:** High. **Expected benefit:** Medium to high. Produces more robust
  estimates across sources but requires several complete training runs.
- [ ] Evaluate test-time augmentation only if error analysis shows a likely gain
  that justifies slower inference and additional ONNX/product complexity.
  **Effort:** High. **Expected benefit:** Low to medium. May recover difficult
  detections, but increases inference time and product complexity.

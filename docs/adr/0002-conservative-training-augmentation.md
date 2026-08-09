# ADR 0002: Use conservative training augmentation

- Status: Accepted
- Date: 2026-08-09

## Context

Lightning channels are thin, high-contrast structures whose visibility can be
destroyed by aggressive image transformations. At the same time, cameras and
weather conditions vary enough that a detector should not memorize one exact
brightness or horizontal composition.

## Decision

Enable conservative augmentation for training samples only: a 50% horizontal
flip with corresponding bounding-box transformation, contrast variation in
`[0.85, 1.15]`, and brightness deltas in `[-0.08, 0.08]`. Validation and test
samples remain unchanged. The training report records whether augmentation was
enabled and its parameters. `--no-augmentation` provides a strict comparison
run.

## Consequences

- The baseline is more robust to moderate camera and exposure variation.
- Geometric annotations remain valid because boxes are transformed with pixels.
- Validation and test metrics remain comparable across experiments.
- More aggressive transformations require a new experiment and validation;
  they must not be added casually because they may erase channel evidence.

## Alternatives considered

- No augmentation: useful as a control, but less robust to distribution shifts.
- Aggressive random crops, rotations and blur: rejected initially because they
  can remove or distort thin lightning channels.
- Test-time augmentation: deferred because it increases inference cost and is
  not needed to establish the first production baseline.

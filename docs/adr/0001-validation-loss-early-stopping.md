# ADR 0001: Stop training from validation loss

- Status: Accepted
- Date: 2026-08-09

## Context

Training runs have a maximum epoch count, but a fixed epoch count can continue
updating a detector after validation performance has stopped improving. The
training repository must select a reproducible checkpoint without using the
test split during model selection.

## Decision

After every training epoch, evaluate the validation split using detection loss.
Stop after `patience` consecutive epochs without an improvement of at least
`min_delta`. Save the checkpoint from the epoch with the lowest validation
loss. The test split remains reserved for final evaluation and release review.

The CLI exposes `--patience` (default `5`) and `--min-delta` (default `0.0`).
Training reports requested and completed epochs, validation-loss history and
the selected best epoch.

## Consequences

- Training can finish before the requested maximum epoch count.
- Checkpoints are selected consistently from validation data rather than from
  the final epoch or training loss alone.
- Every release remains reproducible from the dataset release, seed, CLI
  parameters and training report.
- A validation split is mandatory for training.

## Alternatives considered

- Fixed epochs only: simple, but can overfit and saves no principled best
  checkpoint.
- Select by test loss: rejected because it leaks final evaluation data into
  model selection.
- Select by training loss: rejected because it measures optimization progress,
  not generalization.

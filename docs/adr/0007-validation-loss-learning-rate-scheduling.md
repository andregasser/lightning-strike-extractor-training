# ADR 0007: Schedule learning rate from validation loss

## Status

Accepted

## Context

The detector previously used AdamW with a fixed learning rate of `1e-4` for
every epoch. A fixed rate is a useful control but can keep making steps that are
too large after validation progress plateaus. The existing early-stopping
decision already computes validation loss after every epoch, providing a stable
signal for adapting optimization without consulting the test split.

Scheduler behavior must remain reproducible and must not be hidden behind
framework defaults. It also needs enough time to affect training before early
stopping terminates the run.

## Decision

Training enables PyTorch `ReduceLROnPlateau` by default with:

- validation loss in `min` mode;
- initial AdamW learning rate `1e-4`;
- reduction factor `0.3`;
- scheduler patience two;
- absolute improvement threshold equal to training `min_delta`;
- cooldown zero;
- minimum learning rate `1e-6`;
- epsilon `1e-8`.

The CLI exposes the initial rate, factor, scheduler patience and minimum rate.
`--no-scheduler` preserves a fixed-rate control. With scheduling enabled,
early-stopping patience must exceed scheduler patience by at least two epochs.
This guarantees that a plateau-triggered reduction can be used for at least one
subsequent training epoch before termination.

The schema-version-two training report records every explicit AdamW option,
every scheduler option, the rate used during each epoch, the rate selected for
the next epoch, validation loss, reduction events and the final rate. Scheduler
stepping occurs after validation and before the early-stopping decision.

## Consequences

- Optimization can take smaller steps after validation progress stalls.
- Scheduler and fixed-rate experiments remain directly comparable through one
  explicit CLI switch.
- Runs with incompatible scheduler and early-stopping patience fail before
  loading data or training dependencies.
- The training report grows but contains enough information to reconstruct the
  learning-rate policy and explain every reduction.
- Validation loss now influences both checkpoint selection and learning-rate
  adaptation; the test split remains untouched.

## Alternatives considered

- Keep a fixed learning rate. This remains available as a control but does not
  adapt to convergence.
- Use a fixed step schedule. It is predictable but assumes in advance when
  optimization will plateau across datasets and initialization modes.
- Use cosine annealing or one-cycle scheduling. Both can work well but add more
  schedule-shape assumptions before a trustworthy baseline exists.
- Stop immediately at the first plateau. This gives a reduced rate no chance to
  improve the model and wastes the scheduler signal.

# ADR 0005: Publish dataset composition reports with releases

## Status

Accepted

## Context

Aggregate model metrics can look trustworthy even when a dataset is dominated
by positive images, a small number of sources or one recording domain. The
existing immutable release recorded split totals and source assignments, but it
did not preserve optional camera, recording-condition or rare-case metadata and
did not provide a complete composition artifact for review or experiment
provenance.

## Decision

Every newly built dataset release publishes two checksummed composition files
inside its atomic release operation:

- `reports/dataset-composition.json` is the canonical, schema-versioned report.
- `reports/dataset-composition.md` is a human-readable rendering of that JSON.

The report includes positive and negative image counts, bounding-box counts and
boxes-per-image distribution, per-split summaries, source, camera and recording
condition distributions, rare-case tags and deterministic warnings. Missing
optional metadata is valid and is represented explicitly as `unknown`.
The class-imbalance warning is emitted when positive images account for less
than 20% or more than 80% of all images; these thresholds are recorded in the
JSON report rather than hidden in presentation logic.

Verified COCO image records may carry `camera`, `recording_conditions` and
`rare_cases`. Annotation attributes may carry box-specific `rare_cases`. The
release builder validates and preserves these fields. A training report
references the canonical composition report and its checksum from the immutable
release manifest.

## Consequences

- Dataset imbalance and metadata gaps become visible before training.
- Model runs can identify the exact composition report used for training.
- Existing verified datasets without optional metadata remain valid but produce
  missing-coverage warnings.
- Dataset curators must use stable, consistent tags for useful longitudinal
  comparisons; the schema deliberately does not infer camera or weather labels
  from filenames.
- The Markdown report is derived convenience output and must not replace the
  canonical JSON in automation.

## Alternatives considered

- Generate reports only inside experiment directories. This would duplicate a
  dataset property across runs and weaken the immutable release contract.
- Require all metadata immediately. This would reject current datasets and hide
  useful image, box and source statistics until enrichment was complete.
- Infer conditions automatically from pixels or filenames. This would introduce
  uncertain labels and a model dependency into deterministic release creation.

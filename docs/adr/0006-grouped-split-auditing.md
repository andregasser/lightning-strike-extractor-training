# ADR 0006: Audit hard split groups and visual near-duplicates

## Status

Accepted

## Context

Random image-level splitting is unsafe for video-derived detector datasets.
Adjacent frames, segmented recordings, alternate transcodes and crops can look
nearly identical while carrying different filenames or checksums. If related
images cross train, validation and test boundaries, evaluation measures
memorization and produces unrealistically optimistic metrics.

The release builder already kept one `source_id` in one split and rejected
byte-identical cross-split images. It did not express relationships spanning
multiple source files or detect visually similar content.

## Decision

The verified COCO image contract supports three optional stable group IDs:

- `recording_group_id` joins files from one continuous recording session.
- `event_group_id` joins frames from one lightning event.
- `duplicate_group_id` joins originals, transcodes, crops and known copies.

Together with the required `source_id`, these are hard, indivisible split
groups. Any group assigned to more than one split aborts release publication.
The group namespace is field-specific, so an event and recording may reuse a
text value without becoming related accidentally.

Every release also publishes checksummed `reports/split-audit.json` and
`reports/split-audit.md`. The audit calculates a deterministic 64-bit difference
hash for every decoded image. Hashes within Hamming distance four are joined;
components containing multiple splits are reported with `review-required`
status. Five exact-match hash bands provide complete candidate generation for
the distance threshold without an all-pairs comparison.

Visual matches are warnings rather than hard failures. A curator reviews them
and assigns a stable group ID to confirmed relationships in a subsequent
immutable dataset release. Training reports reference the canonical split-audit
artifact and checksum.

## Consequences

- Known recording and event relationships cannot leak across splits.
- Transcodes and visually near-identical images become reviewable even when
  byte checksums differ.
- Release creation now decodes every image once for hashing and fails safely if
  an image cannot be decoded.
- Existing datasets remain valid without optional group IDs, but the audit
  explicitly reports missing group coverage.
- Difference hashes can flag unrelated dark or saturated scenes; human review
  remains necessary before assigning a hard group.

## Alternatives considered

- Split individual images randomly. This preserves exact percentages but leaks
  adjacent video frames and invalidates evaluation.
- Reject every perceptual-hash match. False positives would make releases
  brittle, especially for visually sparse storm footage.
- Compare every image pair. This is simple but scales quadratically.
- Add video-level acoustic or motion fingerprints immediately. These can help
  with heavily transformed recordings but require original videos and a more
  complex source-level pipeline than the current image release builder.

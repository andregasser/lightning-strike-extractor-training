# Project guidance

## Purpose

This repository owns detector dataset preparation, annotation handoffs,
immutable dataset releases, training, evaluation and ONNX model releases for
the Lightning Strike Extractor. It is independent from the video-analysis CLI.

## Language policy

All repository documentation, source code, code comments, command-line help,
and user-facing messages must be written in English. Keep examples and
technical terminology consistent with this policy.

## Repository split

The project is maintained across two separate Git repositories:

- [`lightning-strike-extractor-training`](https://github.com/andregasser/lightning-strike-extractor-training)
  contains dataset preparation, training, evaluation and ONNX model releases.
- [`lightning-strike-extractor`](https://github.com/andregasser/lightning-strike-extractor)
  contains the production CLI and video-analysis runtime.

The repositories are independent projects with separate dependencies and
lockfiles. They communicate only through versioned file contracts and released
ONNX bundles; this training repo must never import or install the CLI.

## Architecture

- Production training code lives in `src/lse_train/`.
- The CLI entry point is `lse_train.cli:main` and is exposed as
  `lse-train` through `pyproject.toml`.
- Annotation-format adapters and local utilities live in `tools/`.
- Tests live in `tests/` and use `pytest`.
- The training repository must never import `lse` or require the product
  repository to be installed.
- The only product-facing output is a released ONNX bundle with manifest and
  checksums. Training checkpoints and annotation work products stay here.

## Inputs and outputs

- Neutral frame handoffs are the external input from the video-analysis CLI.
- Handoffs are validated before images are copied into annotation campaigns.
- Raw videos are not expected here and must never be copied into the repository.
- Dataset releases are immutable and must never be edited in place.
- Training experiments, checkpoints, caches and generated releases are ignored
  by Git and must never be committed.
- JSON is the canonical format for manifests, provenance and reports.
- Source videos must remain grouped within one train, validation or test split.

## Development commands

Preferred setup and checks:

```bash
uv sync --extra train --extra dev
uv run pytest -q
uv run ruff check .
```

Useful commands:

```bash
uv run lse-train --help
uv run lse-train import-handoff /path/to/frame-export \
  --output campaigns/example
uv run lse-train release /path/to/verified-dataset \
  --release-id example-2026.08 --output releases/example-2026.08
```

## Implementation principles

- Keep handoff and release schemas explicit and versioned.
- Validate paths, image decoding, dimensions, categories, boxes and checksums
  before publishing output.
- Publish campaigns, releases and model bundles atomically; never leave a
  partially valid artifact after a failed operation.
- Prefer deterministic processing and stable ordering.
- Preserve source and frame provenance through every transformation.
- Do not silently resolve conflicting annotations or source assignments.
- Keep training framework details behind training APIs; the ONNX manifest is
  the product contract.
- Treat evaluation metrics and ONNX parity as release gates, not optional logs.

## Architecture decisions

Record every fundamental architecture decision in a dedicated ADR below
`docs/`. This includes repository boundaries, artifact contracts, dataset
schemas, model backends, training architectures, release and promotion rules,
and dependency directions. Each ADR must state status, context, decision,
consequences and relevant alternatives. Update or supersede an ADR explicitly
instead of silently contradicting it in code or documentation.

## Quality and documentation

- Add focused tests for every behavior change, validation rule and edge case.
- Test both successful publication and failure before publication.
- After every substantial change, perform a comprehensive code review before
  committing or handing off the work. Review the complete affected execution
  path rather than only the edited lines, and check architecture and API
  consistency, correctness, failure and interruption behavior, security and
  resource usage, backward compatibility, test coverage, documentation,
  reproducibility and unintended changes elsewhere in the worktree. Resolve
  all actionable findings and rerun the relevant checks after review-driven
  edits.
- Keep README examples runnable against the current code.
- Do not present an unimplemented annotation converter, training path or model
  promotion step as finished functionality.

## Git and commits

- Create a pull request only when explicitly requested.
- Use Conventional Commits 1.0.0 with the form
  `<type>[optional scope][!]: <description>`.
- Every commit needs a concise subject and a non-empty meaningful body that
  explains what changed, why, impact and validation.
- Stage explicit paths when the worktree contains multiple themes.
- Run tests and Ruff before committing.
- Never commit raw images, videos, checkpoints, ONNX artifacts, generated
  releases, caches or experiment outputs.

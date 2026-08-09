# Training workflow for `lse`

This document describes the complete lifecycle of a lightning-channel detector:
from the first video when no dataset exists, through a first ONNX release, to
iterative improvement of an existing model.

The workflow is intentionally split across two repositories:

- [`lightning-strike-extractor`](https://github.com/andregasser/lightning-strike-extractor)
  reads videos, finds candidate frames and exports a neutral frame handoff.
- [`lightning-strike-extractor-training`](https://github.com/andregasser/lightning-strike-extractor-training)
  turns handoffs into verified data, trains and evaluates candidates, and
  produces ONNX bundles.

The repositories must not import one another. The integration boundary is a
versioned file handoff and a released ONNX manifest.

## 1. Prepare both repositories

The training repository needs Python 3.11+, `uv`, and a suitable PyTorch
installation. The inference repository is not installed in the training
environment.

```bash
git clone https://github.com/andregasser/lightning-strike-extractor.git
git clone https://github.com/andregasser/lightning-strike-extractor-training.git

cd lightning-strike-extractor-training
uv sync --extra train --extra dev
uv run lse-train --help
```

Keep raw videos outside Git. A video is read in place by the inference CLI and
is never copied into either repository.

## 2. Start with raw videos and create candidate frames

Run the inference repository against each source video. The workflow has two
different commands here:

- `inspect` is a read-only media check. It probes the input with `ffprobe` and
  reports duration, frame rate, resolution, codecs and audio availability. It
  prints the result as JSON to stdout; it does not persist a report, scan
  frames, create detections or write an analysis run. Redirect stdout yourself
  if you want a standalone snapshot, for example:

  ```bash
  uv run lse inspect /data/storms/camera-a.mp4 > /data/metadata/camera-a.json
  ```

  Example output for a single video:

  ```json
  {
    "path": "/data/storms/camera-a.mp4",
    "name": "camera-a.mp4",
    "duration_seconds": 842.37,
    "size_bytes": 1843920187,
    "bit_rate": 17490000,
    "format": "mov,mp4,m4a,3gp,3g2,mj2",
    "video": {
      "codec_type": "video",
      "codec_name": "hevc",
      "width": 3840,
      "height": 2160,
      "avg_frame_rate": "60000/1001",
      "fps": 59.94005994005994,
      "pix_fmt": "yuv420p"
    },
    "has_audio": true
  }
  ```

  Use it to validate a file and discover safe time ranges before processing.
- `analyze` performs the actual video analysis. It decodes frames, detects
  lightning candidates, ranks events and writes an isolated run with metadata,
  metrics and reviewable outputs. During analysis, the probed source metadata
  is persisted inside the run as `source.json`, alongside run state and result
  files. Use it when you want data that can later be exported into a handoff.

The time range is optional; using a short range is useful for a first smoke
test.

There is no direct data dependency between the commands: `analyze` performs
its own media probing and does not consume an `inspect` JSON file. Therefore,
`inspect` is optional human guidance, while `analyze` is the independent
processing step that creates the run used by the later handoff export.

### How `analyze` ranks frames

The ranking is a deterministic computer-vision pipeline, not a trained model.
It reduces hours of video to a reviewable set of likely channel frames:

1. **Flash detection:** `lse` compares frames with a rolling luminance baseline
   and groups nearby brightness changes into events. Events receive an initial
   rank based on the size and rise of the flash.
2. **Temporal channel response:** for each event, the current grayscale frame
   is compared with a preceding frame. A local ridge response is intersected
   with the temporal difference, so a candidate must be both newly bright and
   line-like. Camera motion is compensated when stabilization is enabled.
3. **Channel geometry:** the response is thresholded, lightly closed,
   skeletonized and split into connected components. The strongest component
   contributes line count, channel length, channel strength, branch count,
   thickness, luminance and a geometry score. Dense frame-wide edges and large
   bright areas are penalized because they are typical of motion, clouds or
   exposure flashes.
4. **Frame quality:** the principal quality score rewards long, bright,
   branched, thin channels. In simplified form it is proportional to:

   ```text
   channel_length * (channel_luminance / 10)^2
   * (1 + 0.15 * min(branch_points, 10))
   / max(channel_thickness, 1)
   ```

   The geometry score and quality score are related but distinct: geometry
   measures response/component evidence, while quality emphasizes how clearly
   the component resembles a visible channel.
5. **Multi-frame support:** when enabled, nearby candidate masks are dilated
   slightly and compared. Spatial overlap within the configured time window
   adds a bounded support bonus. This favors a channel that develops over
   adjacent frames without allowing temporal overlap to invent geometry where
   none exists.
6. **Event winner selection:** candidates are grouped by `event_id`. The
   winner is selected by `(frame_quality, multiframe_support,
   geometry_score)`, with multi-frame quality used for final event ordering.
   Strict geometry, length, line-count, strength and thickness gates reject
   implausible candidates before export.

The resulting metrics are written to the analysis run, primarily:

```text
runs/videos/<video-run-id>/
├── source.json
├── run.json
├── results/
│   ├── candidates.json
│   └── candidates.csv
└── events/
```

`candidates.json` is the authoritative machine-readable ranking output. The
later handoff exporter uses it to select one strong representative per event
and adds neighboring context frames. Thresholds remain configurable because
camera noise, exposure, frame rate and weather vary; changing them is an
inference-pipeline experiment and should be recorded separately from model
training.

```bash
cd ../lightning-strike-extractor
uv sync --extra dev

uv run lse inspect /data/storms/2026-08-09/camera-a.mp4
uv run lse analyze /data/storms/2026-08-09/camera-a.mp4 \
  --start 0 --end 600 \
  --output runs
```

For a batch of videos, use the batch commands documented in the inference
repository. The important output is an analysis run containing candidate
frames, source metadata and provenance. The source video remains external.

## 3. Export a neutral frame handoff

### What “handoff” means

The handoff is the file-based delivery package between the two repositories.
It is not a Python import, a shared database or the final training dataset.
`lse` creates it; `lse-train` validates and imports it.

Each handoff contains selected image files and a `manifest.json`. The manifest
records the schema version, source/video identity, frame number, timestamp,
analysis metrics, relative file name and SHA-256 checksum for every image. This
lets the training repository verify that it received exactly the frames that
the inference repository exported, without needing access to the original
video or to the `lse` Python package.

Conceptually, the boundary looks like this:

```text
raw video ──> lse ──> handoff directory ──> lse-train ──> annotations/dataset/model
                         manifest.json
                         images/
```

The handoff is therefore an immutable transfer snapshot. If the source video
or frame selection changes, create a new handoff instead of modifying the old
one. Annotation work and verified dataset files are created after the handoff
has been imported.

Export selected candidates plus temporal context. This command produces a
portable directory with `manifest.json`, images, source IDs, frame numbers,
timestamps, candidate metrics and SHA-256 checksums.

```bash
uv run python -m lse.dataset_export runs \
  --output /data/handoffs/lightning-2026-08-09-camera-a \
  --max-events-per-video 200 \
  --context-frames 2
```

The handoff is the only training input copied from the inference repository.
Before transferring it, inspect `manifest.json` and verify that the image files
are present. Do not edit image pixels or provenance after export.

## 4. Import the handoff as an annotation campaign

In the training repository, validate the handoff and create a campaign:

```bash
cd ../lightning-strike-extractor-training

uv run lse-train import-handoff \
  /data/handoffs/lightning-2026-08-09-camera-a \
  --output campaigns/lightning-2026-08-09-camera-a
```

The importer rejects unsafe paths, missing files, checksum mismatches,
duplicate images, undecodable images and conflicting task names. It creates:

```text
campaigns/lightning-2026-08-09-camera-a/
├── manifest.json
├── annotation/
│   ├── tasks.json
│   └── label-config.xml
└── serve/images/
```

The current campaign format is designed for rectangle annotation of the single
class `lightning_channel`. Annotate every image, including negatives. Draw a
tight box around each visible channel; do not label a bright flash without a
visible channel. Preserve the source ID and frame provenance in every exported
annotation.

## 5. Produce a verified dataset

After annotation, create a verified COCO-style campaign with this layout:

```text
verified-dataset/
├── manifest.json
├── annotations/
│   ├── instances_train.json
│   ├── instances_validation.json
│   └── instances_test.json
└── images/
    ├── train/
    ├── validation/
    └── test/
```

The annotation export must contain the category `lightning_channel`, image
dimensions, bounding boxes in `[x, y, width, height]` format, source IDs and
checksums. Keep all frames from one source video in exactly one split. Never
split adjacent frames from the same source across train and test.

The current CLI release command validates one or more already verified
campaigns and creates an immutable release:

```bash
uv run lse-train release \
  /data/verified/lightning-v1 \
  --release-id lightning-2026.08.1 \
  --output releases/datasets/lightning-2026.08.1
```

The release contains deterministic image names, split annotations, source
assignments, campaign provenance and checksums. If the data changes, create a
new release ID; never edit a published release in place.

## 6. Train the first baseline

The current baseline is Faster R-CNN with a MobileNet backbone. It is a
reference point for reproducible comparisons, not a claim that it is the final
architecture.

```bash
uv run lse-train train \
  releases/datasets/lightning-2026.08.1 \
  --output experiments/2026-08-09-fasterrcnn-mobilenet \
  --epochs 10 \
  --seed 17
```

The command trains only on the `train` split and writes:

```text
experiments/2026-08-09-fasterrcnn-mobilenet/
├── checkpoint.pt
└── training.json
```

`training.json` records the dataset release, architecture, epoch count, seed
and loss history. Treat the checkpoint and report as one experiment; do not
rename or detach them from their dataset release.

## 7. Evaluate before exporting

Evaluate on validation data while tuning. Use the test split only for a final,
unbiased report.

```bash
uv run lse-train evaluate \
  releases/datasets/lightning-2026.08.1 \
  experiments/2026-08-09-fasterrcnn-mobilenet/checkpoint.pt \
  --split validation \
  --output experiments/2026-08-09-fasterrcnn-mobilenet/validation.json
```

The report includes true positives, false positives, false negatives,
precision and recall. The current defaults are confidence `0.25` and IoU `0.5`.
Record the thresholds when comparing models. A model is not better merely
because recall increased: inspect false positives, false negatives and source-
isolated results as well.

## 8. Export and verify an ONNX release

Only export a candidate after reviewing its validation report and checking that
the dataset release and training report are the intended inputs:

```bash
uv run lse-train export-onnx \
  experiments/2026-08-09-fasterrcnn-mobilenet/checkpoint.pt \
  --training-report experiments/2026-08-09-fasterrcnn-mobilenet/training.json \
  --evaluation-report experiments/2026-08-09-fasterrcnn-mobilenet/validation.json \
  --version 0.1.0 \
  --output releases/models/lightning-channel-detector-0.1.0
```

The exporter checks the ONNX graph, runs ONNX Runtime and compares its outputs
with PyTorch outputs. A successful release contains:

```text
releases/models/lightning-channel-detector-0.1.0/
├── model.onnx
├── manifest.json
└── checksums.json
```

The manifest fixes the input tensor, preprocessing, class schema, output
tensors, thresholds, ONNX opset, runtime requirement, dataset provenance and
artifact hash. Copy or promote this complete bundle into the inference
repository; never copy a raw checkpoint into production.

## 9. Improve an existing model safely

Model improvement is a controlled experiment, not a replacement of the
production file in place.

### 9.1 Identify the failure mode

Collect examples from real inference runs and classify them:

- false negatives: visible channels the model missed;
- false positives: cloud edges, exposure flashes, lamps or compression noise;
- localization errors: the channel was found but the box is too large or small;
- confidence errors: correct detections ranked below the operating threshold;
- distribution gaps: a camera, exposure, weather or resolution not represented
  in training data.

Use the inference CLI's review outputs and frame handoff provenance to retain
the source video, timestamp, run ID and original frame identity for every case.
Do not add only the frames that look easy; include hard negatives and nearby
context frames.

### 9.2 Add a new campaign and dataset release

Export new failure cases from the inference repository, then import them into a
new campaign. Combine the new verified campaign with the previous verified data
when building the next release:

```bash
uv run lse-train import-handoff \
  /data/handoffs/model-review-2026-08-15 \
  --output campaigns/model-review-2026-08-15

uv run lse-train release \
  /data/verified/lightning-v1 \
  /data/verified/model-review-2026-08-15 \
  --release-id lightning-2026.08.2 \
  --output releases/datasets/lightning-2026.08.2
```

Do not silently relabel old data. Record changed labels, new sources and the
reason for the release in its manifest or accompanying experiment notes.

### 9.3 Retrain under controlled conditions

Keep the architecture, seed and evaluation protocol fixed when measuring a
dataset change. Change one major variable at a time:

```bash
uv run lse-train train \
  releases/datasets/lightning-2026.08.2 \
  --output experiments/2026-08-15-hard-negatives \
  --epochs 10 \
  --seed 17

uv run lse-train evaluate \
  releases/datasets/lightning-2026.08.2 \
  experiments/2026-08-15-hard-negatives/checkpoint.pt \
  --split validation \
  --output experiments/2026-08-15-hard-negatives/validation.json
```

Compare the new report with the previous report on the same validation/test
policy. Keep a candidate only if it improves the target behavior without an
unacceptable regression on other source groups.

### 9.4 Export, promote and monitor

Export the winning checkpoint with a new model version, run ONNX parity checks,
and promote the complete bundle. Keep the previous model available for rollback.
After promotion, sample new inference runs and feed confirmed failures back
into the next campaign. This creates an evaluation loop rather than repeatedly
training on ad-hoc screenshots.

## 10. Reproducibility checklist

Before calling a model release ready, record:

1. dataset release ID and source-to-split assignments;
2. annotation schema and label policy;
3. architecture, dependency lockfile and training seed;
4. epoch count and training report;
5. validation and final test precision/recall with thresholds;
6. ONNX opset, preprocessing and parity result;
7. model version, SHA-256 checksums and promotion target;
8. known failure cases and rollback version.

Run the repository checks before committing code or documentation:

```bash
uv run pytest -q
uv run ruff check .
```

This workflow describes the current implementation. If a future change adds
augmentation, hyperparameter search, a new architecture or an automated
annotation conversion command, document that change in an ADR and update this
guide with the exact reproducible commands.

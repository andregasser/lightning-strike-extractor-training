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

## Best practices for video source storage

Raw videos are the source of truth for future dataset growth. Store them in a
dedicated external data volume or object store, not in Git, `data/` inside this
repository, experiment folders or model releases. Both repositories should
contain only code, manifests, documentation and small test fixtures.

Use a stable, date- and source-oriented layout. For example:

```text
/data/lightning/
├── sources/
│   ├── 2026/
│   │   └── 2026-08-09/
│   │       ├── camera-a/
│   │       │   ├── camera-a_2026-08-09T214500Z.mp4
│   │       │   └── camera-a_2026-08-09T220000Z.mp4
│   │       └── camera-b/
│   │           └── camera-b_2026-08-09T214500Z.mp4
│   └── source-catalog.json
├── handoffs/
├── verified-datasets/
└── backups/
```

Follow these rules:

1. **Keep source files immutable.** Never overwrite, trim, re-encode or rename
   a video after it has been analyzed. If a corrected or transcoded copy is
   needed, create a new source record and a new source ID.
2. **Use stable source identity.** Record a source ID independent of the local
   path, for example a hash or an assigned identifier such as
   `camera-a-2026-08-09-214500z`. Preserve the original filename, capture time,
   camera, location (when appropriate), duration, codec, resolution and FPS in
   `source-catalog.json` or an equivalent external catalog.
3. **Separate storage from generated data.** Keep raw videos under `sources/`,
   exported handoffs under `handoffs/`, verified dataset inputs under
   `verified-datasets/` and backups under `backups/`. Do not mix generated
   frames back into the raw source directory.
4. **Verify integrity before analysis.** Record a SHA-256 checksum and file
   size when a video enters the catalog. Recheck the checksum after transfers
   or restores. The analysis run stores the probed source metadata and source
   identity, while the handoff stores checksums for exported frames.
5. **Use local paths only as configuration.** The absolute path passed to
   `lse` may differ between machines. The source ID and provenance in the run
   and handoff must remain stable so that a dataset release can be reproduced
   without relying on one workstation's directory layout.
6. **Back up before annotation.** Keep at least one independent, read-only or
   versioned backup of the raw videos and the source catalog. Dataset releases
   and handoffs can be regenerated from the source videos, but only if the
   original bytes remain available.

Before analyzing a new source, verify that the file is present and readable:

```bash
uv run lse inspect /data/lightning/sources/2026/2026-08-09/camera-a/camera-a_2026-08-09T214500Z.mp4
```

The path above is intentionally local to the machine running `lse`; it is not
committed to either repository. After analysis, keep the resulting run and
handoff IDs in the external source catalog or project tracking system.

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
  files. The run root defaults to `runs`, so the normal command does not need an
  explicit output path:

  ```bash
  uv run lse analyze /data/storms/2026-08-09/camera-a.mp4 \
    --start 0 --end 600
  ```

  This creates a run below `runs/videos/<video-run-id>/`. Use `--output` only
  when you intentionally want to place the complete run tree elsewhere. Use
  `analyze` when you want data that can later be exported into a handoff. The
  time range is optional; using a short range is useful for a first smoke test.

  For a batch of videos, pass a directory. Add `--recursive` when nested
  directories should be searched:

  ```bash
  uv run lse analyze /data/storms/2026-08-09 \
    --recursive
  ```

  The command creates one video run below `runs/videos/` per discovered video
  and one orchestration run below `runs/batches/` for the complete batch.
  Leave `--max-events` unset for complete processing; its default value `0`
  means that no event limit is applied. Use a positive value only for a
  deliberately bounded exploratory run.

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
uv run lse dataset-export runs \
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

The campaign paths have deliberately separate responsibilities:

| Path | Purpose | Lifecycle |
| --- | --- | --- |
| `manifest.json` | Campaign metadata, source handoff reference, task count, source distribution, image dimensions and checksums. | Immutable campaign record; keep it with the campaign. |
| `annotation/tasks.json` | Label Studio task definitions. Each task points to an image URL and carries source/frame provenance. | Input to Label Studio; do not edit manually. |
| `annotation/label-config.xml` | Label Studio labeling interface for the `lightning_channel` rectangle class. | Versioned configuration for the campaign. |
| `serve/images/` | Validated copies of the handoff images served to Label Studio over HTTP. | Disposable/recreatable serving area; never treat it as the raw source. |

The image copies under `serve/images/` are intentionally separate from
`annotation/`. Label Studio reads pixels from the HTTP server, while the task
file stores URLs and provenance. The campaign manifest remains the integrity
record. If the serving directory is lost, recreate the campaign from the
original handoff rather than copying files back from an annotation export.

### Label Studio preparation

The campaign output is prepared specifically for Label Studio. The importer
does not upload anything to a hosted annotation service and does not require
the `lse` package. It creates a self-contained task package that can be served
locally or copied to the annotation workstation:

- `serve/images/` contains the validated image copies that Label Studio must
  load. The original handoff remains unchanged.
- `annotation/tasks.json` contains one task per image. Each task includes the
  image URL, stable `source_id`, original file name and complete frame
  provenance. The URLs use the `--image-base-url` value passed to
  `import-handoff` (by default `http://localhost:8001/images`).
- `annotation/label-config.xml` defines the single rectangle class
  `lightning_channel` and instructs the annotator to draw a tight box around
  every visible channel.
- `manifest.json` records the campaign inputs, checksums, dimensions, task
  count and source distribution.

Serve the prepared images from the campaign directory before importing the
tasks into Label Studio. For example, from the campaign root:

```bash
cd campaigns/lightning-2026-08-09-camera-a
python3 -m http.server 8001 --directory serve
```

In Label Studio, create a project with the contents of
`annotation/label-config.xml` as its labeling interface, then import
`annotation/tasks.json` as the task file. The project must be able to resolve
the image URLs from the running server. If the server uses another host or
port, pass the matching URL when creating the campaign:

```bash
uv run lse-train import-handoff \
  /data/handoffs/lightning-2026-08-09-camera-a \
  --output campaigns/lightning-2026-08-09-camera-a \
  --image-base-url http://annotation-host:8001/images
```

Label every task, including negative images. Export the completed annotations
from Label Studio and convert or normalize them into the verified COCO-style
campaign layout described in the next section. Keep the exported annotation
file together with the campaign manifest and preserve the source IDs and frame
provenance; do not train directly from the raw Label Studio export.

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
checksums. Images may additionally contain a non-empty `camera` string and
unique string lists named `recording_conditions` and `rare_cases`. Annotation
attributes may contain a unique `rare_cases` string list for box-specific
cases. Keep all frames from one source video in exactly one split. Never split
adjacent frames from the same source across train and test.

The current CLI release command validates one or more already verified
campaigns and creates an immutable release:

```bash
uv run lse-train release \
  /data/verified/lightning-v1 \
  --release-id lightning-2026.08.1 \
  --output releases/datasets/lightning-2026.08.1
```

The release contains deterministic image names, split annotations, source
assignments, campaign provenance and checksums. During the same atomic release
operation, it generates `reports/dataset-composition.json` and
`reports/dataset-composition.md`. The JSON report is canonical; the Markdown
companion summarizes positive and negative images, box counts, split and source
distributions, cameras, recording conditions, rare cases and warnings for
class imbalance or missing metadata. Review it before starting training.
Training records the report path and SHA-256 from the release manifest in
`training.json`. If the data changes, create a new release ID; never edit a
published release in place.

## 6. Train the first baseline

The detector is Faster R-CNN with a ResNet-50 FPN V2 backbone. Training uses
versioned COCO detector weights by default, then replaces the COCO classifier
with a background-plus-`lightning_channel` predictor. ImageNet-backbone and
random initialization remain available as controlled comparison modes.

```bash
uv run lse-train train \
  releases/datasets/lightning-2026.08.1 \
  --output experiments/2026-08-09-fasterrcnn-resnet50-fpn-v2 \
  --epochs 10 \
  --seed 17 \
  --patience 5
```

The command trains only on the `train` split and writes:

```text
experiments/2026-08-09-fasterrcnn-resnet50-fpn-v2/
├── checkpoint.pt
└── training.json
```

Training evaluates the validation split after every epoch. If validation loss
does not improve by at least `--min-delta` for `--patience` consecutive epochs,
training stops early. The saved `checkpoint.pt` is always the best validation
checkpoint, not necessarily the final epoch. Set `--patience 0` only when you
intentionally want to stop at the first non-improving epoch; use a larger value
for noisy validation curves.

`training.json` records the dataset release, the composition-report path and
checksum, architecture, exact initialization mode and weight identifiers,
device, requested and completed epochs, seed, training losses, validation
losses and early-stopping metadata. Treat the checkpoint and report as one
experiment; do not rename or detach them from their dataset release.

Run controlled initialization comparisons with otherwise identical settings:

```bash
uv run lse-train train releases/datasets/lightning-2026.08.1 \
  --output experiments/imagenet-backbone \
  --initialization imagenet-backbone

uv run lse-train train releases/datasets/lightning-2026.08.1 \
  --output experiments/random-control \
  --initialization random
```

The default COCO mode may download official Torchvision weights on first use.
Subsequent runs use the normal PyTorch cache. Training selects CUDA first,
Apple Metal through MPS second, and CPU otherwise.

Training augmentation is enabled by default. The current conservative pipeline
randomly mirrors training images horizontally and applies small contrast and
brightness changes. Bounding boxes are mirrored with the image; validation and
test images are never augmented. Use `--no-augmentation` only for a strict
non-augmented comparison run:

```bash
uv run lse-train train \
  releases/datasets/lightning-2026.08.1 \
  --output experiments/no-augmentation \
  --epochs 10 \
  --no-augmentation
```

Augmentation changes the experiment definition, so compare augmented and
non-augmented runs using the same dataset release, seed, validation split and
evaluation thresholds.

## 7. Evaluate before exporting

Evaluate on validation data while tuning. Use the test split only for a final,
unbiased report.

```bash
uv run lse-train evaluate \
  releases/datasets/lightning-2026.08.1 \
  experiments/2026-08-09-fasterrcnn-resnet50-fpn-v2/checkpoint.pt \
  --split validation \
  --score-threshold 0.25 \
  --iou-threshold 0.5 \
  --output experiments/2026-08-09-fasterrcnn-resnet50-fpn-v2/validation.json
```

The JSON report includes true positives, false positives, false negatives,
precision and recall at the selected operating point. It also includes AP at
IoU 0.5, COCO-style mAP averaged across IoU thresholds 0.5–0.95, per-IoU AP and
the full IoU-0.5 precision-recall curve. A sibling
`validation-pr-curve.csv` file contains the curve in tabular form. The current
operating-point defaults are confidence `0.25` and IoU `0.5`.

AP and mAP use every model score and therefore compare ranking quality without
depending on the selected operating-point confidence. Do not interpret one
metric in isolation: inspect AP, strict-IoU mAP, false positives, false
negatives, the precision-recall tradeoff and source-isolated results before
deciding that a candidate is better.

## 8. Export and verify an ONNX release

Only export a candidate after reviewing its validation report and checking that
the dataset release and training report are the intended inputs:

```bash
uv run lse-train export-onnx \
  experiments/2026-08-09-fasterrcnn-resnet50-fpn-v2/checkpoint.pt \
  --training-report experiments/2026-08-09-fasterrcnn-resnet50-fpn-v2/training.json \
  --evaluation-report experiments/2026-08-09-fasterrcnn-resnet50-fpn-v2/validation.json \
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

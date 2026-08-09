<p align="center">
  <img src="docs/assets/lightning-extractor-training-hero.png" alt="Storm lightning with an overlaid detection and dataset pipeline" width="100%">
</p>

# Lightning Strike Extractor Training

Independent dataset and detector development for the
[Lightning Strike Extractor CLI](https://github.com/andregasser/lightning-strike-extractor).

This repository owns the complete model lifecycle: importing neutral frame
handoffs, preparing annotation campaigns, creating immutable dataset releases,
training and evaluating detector candidates, and exporting verified ONNX model
bundles. It is deliberately independent from the video-analysis CLI and must
never import or install `lse`.

The [product repository](https://github.com/andregasser/lightning-strike-extractor)
consumes one thing from this project: a released ONNX bundle with a manifest and
checksums. Training checkpoints and framework-specific details never enter the
product distribution.

Together, the repositories form a deliberately separated pipeline: the CLI
analyzes videos and creates provenance-rich frame handoffs; this repository
turns those handoffs into verified datasets, trained models, and versioned ONNX
releases for the CLI.

## Boundary and data flow

```text
lightning-strike-extractor             lightning-strike-extractor-training
────────────────────────────           ───────────────────────────
raw video                              frame handoff
    │                                  │
    ├─ lightning analyze                ├─ import-handoff
    └─ frame/provenance export ────────┼─ annotation campaign
                                       ├─ verified dataset release
                                       ├─ train + evaluate
                                       └─ ONNX release ────────────┐
                                                                   │
                         product model promotion ◄────────────────┘
```

The handoff is a file contract, not a Python API. The source video remains in
the CLI environment; this project receives selected images, source identity,
timestamps, candidate metrics and SHA-256 checksums.

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a suitable
PyTorch installation for the selected training hardware.

```bash
git clone git@github.com:andregasser/lightning-strike-extractor-training.git
cd lightning-strike-extractor-training
uv sync --extra train --extra dev
```

The product CLI is not required and must not be installed in this environment.

## Import a CLI frame handoff

Create a neutral handoff in the video repository:

```bash
uv run lse dataset-export runs \
  --output /path/to/frame-export
```

Import it here:

```bash
uv run lse-train import-handoff /path/to/frame-export \
  --output campaigns/storm-2026-08
```

The importer validates schema version, safe relative paths, source identity,
image decoding, duplicate filenames and hashes, and SHA-256 checksums. It
atomically creates annotation tasks, a label configuration, served images and a
campaign manifest. Provenance remains attached to every task.

## Annotation and verified data

Annotate every task, including true negatives. A visible channel uses the
single class `lightning_channel`; an image without a channel remains a valid
negative example.

The handoff importer stops at an annotation campaign. Conversion from completed
annotation exports into a verified, source-grouped dataset is a separate step in
this repository. Do not train directly from an imported campaign or unreviewed
proposals.

The release builder expects:

```text
verified-dataset/
├── manifest.json
├── annotations/
│   ├── instances_train.json
│   ├── instances_validation.json
│   └── instances_test.json
└── images/{train,validation,test}/
```

Complete source videos must remain in exactly one split. Splitting adjacent
frames from the same source across train and test produces misleading metrics.

## Build an immutable dataset release

```bash
uv run lse-train release /path/to/verified-dataset \
  --release-id lightning-2026.08.1 \
  --output releases/lightning-2026.08.1
```

The release builder accepts only human-verified annotations, validates image
dimensions, paths and boxes, deduplicates identical image content by SHA-256,
rejects conflicting annotations and source/split conflicts, refuses to overwrite
an existing release, and writes source assignments and output checksums. Every
new release also contains `reports/dataset-composition.json` as the canonical
composition record and `reports/dataset-composition.md` as its human-readable
summary. These reports cover positive and negative images, boxes per image,
sources, cameras, recording conditions, rare-case tags and coverage warnings.

Verified COCO images may provide a non-empty `camera` string plus unique
`recording_conditions` and `rare_cases` string lists. Annotation attributes may
also provide a unique `rare_cases` string list for box-specific cases. Missing
optional metadata remains valid and is reported as unknown coverage.

Split integrity is checked at the same time. `source_id`,
`recording_group_id`, `event_group_id` and `duplicate_group_id` are treated as
indivisible groups and a cross-split assignment aborts publication. The release
also contains checksummed `reports/split-audit.json` and
`reports/split-audit.md` files. A 64-bit difference hash finds visually similar
images across splits; these candidates require review because perceptual
similarity is evidence, not proof, of duplicate recordings.

Releases are immutable inputs to experiments. Create a new release ID when data
changes; never edit an existing release in place.

## Train and evaluate

The current implementation uses Faster R-CNN with a ResNet-50 FPN V2 backbone.
COCO detector pretraining is the default; the COCO classification predictor is
replaced with the project-specific background-plus-`lightning_channel` head.
ImageNet-backbone and random initialization remain explicit comparison modes.
See `docs/adr/0004-resnet50-fpn-v2-detector.md` and the prioritized
`docs/TRAINING_TODO.md` roadmap.

```bash
uv run lse-train train releases/lightning-2026.08.1 \
  --output experiments/baseline --epochs 10 --seed 17 --patience 5

uv run lse-train evaluate releases/lightning-2026.08.1 \
  experiments/baseline/checkpoint.pt --split validation \
  --output experiments/baseline/validation.json
```

Evaluation reports true positives, false positives, false negatives, precision
and recall at a documented operating point, plus AP at IoU 0.5, COCO-style mAP
across IoU 0.5–0.95 and a precision-recall curve in JSON and CSV form. Training
evaluates validation loss after every epoch, stops after `--patience`
non-improving epochs, and saves the best validation checkpoint. Compare models
on the same immutable release and source-isolated splits before promotion.
The training report references the checksummed dataset-composition and split
audit JSON files from the release so experiment results retain their data and
split-integrity provenance.

Training uses a validation-loss-driven `ReduceLROnPlateau` scheduler by default:
initial learning rate `1e-4`, reduction factor `0.3`, patience two and minimum
rate `1e-6`. `training.json` records the complete AdamW and scheduler
configuration, per-epoch learning-rate history, reduction events and final
rate. Use `--no-scheduler` for a controlled fixed-rate comparison. When the
scheduler is enabled, early-stopping patience must exceed scheduler patience by
at least two epochs so a reduced rate can affect a later epoch.

The schema-version-three training report also records performance evidence for
each epoch and the complete run: elapsed training and validation time,
DataLoader wait, device transfer, optimization, image and batch throughput,
process peak resident memory and backend-specific device memory. CUDA uses
resettable PyTorch peak counters. MPS memory is sampled at synchronized phase
boundaries because PyTorch does not expose an equivalent peak counter. Reliable
device-utilization percentages are reported as unavailable rather than
estimated without a trustworthy backend API.

Use `--initialization imagenet-backbone` or `--initialization random` for a
controlled transfer-learning comparison. The first COCO-pretrained run may
download official versioned Torchvision weights; later runs use the PyTorch
cache. Training prefers CUDA, then Apple Metal through MPS, and otherwise CPU.

## Export and verify ONNX

```bash
uv run lse-train export-onnx experiments/baseline/checkpoint.pt \
  --training-report experiments/baseline/training.json \
  --evaluation-report experiments/baseline/validation.json \
  --version 0.1.0 --output releases/model-0.1.0
```

The exporter validates the graph, runs ONNX Runtime, compares ONNX outputs with
PyTorch outputs, and writes this bundle atomically:

```text
releases/model-0.1.0/
├── model.onnx
├── manifest.json
└── checksums.json
```

The manifest fixes preprocessing, tensor names, class schema, thresholds,
opset, runtime compatibility, dataset provenance and artifact hash.

## Promote a model to the product

Promotion is explicit. After reviewing evaluation and ONNX parity, copy the
released artifact into the product repository and update its manifest. The
product runtime then validates the checksum and tensor contract. The training repository
never pushes a model directly into a production checkout, and the product never
downloads research checkpoints implicitly.

## Repository layout

```text
src/lse_train/
├── handoff.py          # frame handoff validation and campaigns
├── coco.py             # verified dataset validation
├── release.py          # immutable dataset releases
├── model.py            # detector architecture, initialization and device selection
├── training.py         # baseline training
├── evaluation.py       # candidate evaluation
├── onnx_release.py     # export and parity checks
└── cli.py              # lse-train entry point
tools/                  # annotation-format adapters and local utilities
tests/                  # deterministic training tests
```

## Development checks

```bash
uv sync --extra train --extra dev
uv run pytest -q
uv run ruff check .
```

Before committing substantial changes, review the complete affected path:
artifact schemas, failure behavior, resource usage, input-path security,
reproducibility, tests and documentation.

## Status

The repository contains the independent lifecycle infrastructure and a baseline
detector. No model is production-ready until it has been evaluated on a real,
source-isolated test set and its ONNX bundle has passed parity checks.

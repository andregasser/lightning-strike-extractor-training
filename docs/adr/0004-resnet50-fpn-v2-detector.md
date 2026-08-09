# ADR 0004: Use Faster R-CNN with ResNet-50 FPN V2

## Status

Accepted

## Context

The initial detector used the low-resolution
`fasterrcnn_mobilenet_v3_large_320_fpn` architecture with random weights.
MobileNet and the 320-pixel training scale prioritize mobile efficiency, but
mobile deployment is not a project requirement. Thin lightning channels may
lose important spatial evidence at low resolution, and the project prefers
detection quality over minimum model size.

The development environment includes macOS. PyTorch can execute ResNet-based
Torchvision detection models on CPU and can use Apple Metal through MPS when
the installed PyTorch version and required operations support it. The product
boundary remains ONNX, so macOS training support does not change the released
runtime contract.

## Decision

Use `fasterrcnn_resnet50_fpn_v2` as the single detector architecture for
training, evaluation and ONNX export. Use versioned Torchvision COCO detector
weights by default, replace the COCO classification predictor with a two-class
background-plus-`lightning_channel` predictor, and fine-tune the complete
network.

Retain two explicit comparison modes:

- `imagenet-backbone` initializes only ResNet-50 with
  `ResNet50_Weights.IMAGENET1K_V2`;
- `random` initializes the complete detector without pretrained weights.

Training reports and model manifests record the architecture, initialization
mode and exact weight identifiers. Device selection prefers CUDA, then MPS,
and otherwise CPU. Evaluation and export construct the architecture without
external weights before loading the project checkpoint, avoiding unnecessary
downloads and preventing pretrained state from leaking into evaluation.

## Consequences

- The model has substantially more capacity and compute cost than the MobileNet
  320 baseline.
- COCO pretraining transfers feature extraction, region proposal and box
  regression behavior; the lightning classifier remains project-specific.
- Apple Silicon can accelerate supported operations through MPS, but the full
  training and export paths still require practical compatibility tests.
- The larger model increases training time, memory use, ONNX artifact size and
  product inference cost.
- Model promotion remains gated on source-isolated evaluation, ONNX parity and
  measured runtime behavior.

## Alternatives considered

### Keep the MobileNet 320 detector

This minimizes compute and artifact size but optimizes for a mobile constraint
the project does not have and may discard thin-channel detail.

### Use the high-resolution MobileNet detector

This reduces migration cost and preserves more spatial detail, but retains an
efficiency-oriented backbone when quality is the primary requirement.

### Switch immediately to RetinaNet or FCOS

Both are credible future experiments, but changing both the backbone and the
detection method at once would make the first comparison harder to interpret
and would expand ONNX integration risk.

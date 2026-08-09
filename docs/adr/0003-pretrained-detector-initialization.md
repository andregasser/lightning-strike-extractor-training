# ADR 0003: Compare pretrained detector initialization strategies

## Status

Superseded by ADR 0004

## Context

The Faster R-CNN/MobileNet V3/FPN baseline currently initializes the complete
network randomly. The first verified lightning-channel dataset will be small
relative to general-purpose image and object-detection datasets. Learning both
basic visual features and object-localization behavior from that dataset alone
would make training slower, more seed-sensitive and more prone to fitting
source-specific artifacts.

Torchvision provides two relevant transfer-learning boundaries for
`fasterrcnn_mobilenet_v3_large_320_fpn`:

1. ImageNet-pretrained MobileNet V3 weights initialize the feature-extracting
   backbone while FPN, region proposal and detection components remain newly
   initialized.
2. COCO-pretrained Faster R-CNN weights initialize the backbone, FPN, region
   proposal network and box-localization behavior. The COCO classification head
   must be replaced with a new background-plus-`lightning_channel` head.

These are alternative initialization experiments, not cumulative training
steps. COCO detector weights already represent an end-to-end detector
pretraining path; applying separate ImageNet weights afterward would overwrite
part of that learned state.

## Decision

The training CLI will expose three explicit initialization modes:

- `random`: no pretrained weights, retained as the control baseline;
- `imagenet-backbone`: initialize MobileNet V3 from the versioned Torchvision
  ImageNet weights and initialize the detection-specific components anew;
- `coco-detector`: initialize the complete detector from the versioned
  Torchvision COCO weights and replace its classification predictor for the
  project class schema.

The first model-selection experiment will run all three modes against the same
immutable dataset release, source-isolated splits, seed, augmentation,
optimizer settings and training budget. The COCO detector mode is the preferred
first candidate because it transfers object proposal and box-regression
behavior in addition to image features. It is not promoted by assumption: the
validation and test results must outperform or otherwise justify it against the
ImageNet-only and random controls.

Every training report and released model manifest must record the mode and the
exact Torchvision weight identifiers. Weight downloads must use Torchvision's
versioned weight enums and normal cache behavior rather than unversioned URLs.

## Consequences

- Initial training may converge faster and more consistently on limited data.
- The comparison separates the value of general visual features from the value
  of detector-level transfer learning.
- The initial weight files become reproducibility inputs and must be identified
  in reports.
- The first use may require a network download; subsequent runs may use the
  local PyTorch cache.
- COCO categories do not include lightning channels, so the final classifier
  remains project-specific and requires verified lightning annotations.
- Pretraining can introduce source-domain biases, making source-grouped
  evaluation and hard-negative review mandatory.

## Alternatives considered

### Keep random initialization only

This remains a useful control but wastes established general visual and
object-localization features when the project dataset is comparatively small.

### Use only ImageNet backbone weights

This is simpler and remains part of the comparison, but it does not transfer
region-proposal or box-regression behavior.

### Use only COCO detector weights without a control

This would reduce experiment cost, but it would prevent measuring whether the
domain difference between everyday COCO objects and lightning channels makes
detector-level transfer helpful.

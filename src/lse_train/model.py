from __future__ import annotations

from typing import Any, Literal

ARCHITECTURE = "fasterrcnn_resnet50_fpn_v2"
INITIALIZATIONS = ("random", "imagenet-backbone", "coco-detector")
Initialization = Literal["random", "imagenet-backbone", "coco-detector"]


def build_detector(
    torchvision: Any, *, initialization: Initialization = "random"
) -> tuple[Any, dict[str, str | None]]:
    """Build the project detector and return reproducible initialization metadata."""
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"Unsupported model initialization: {initialization}")
    detection = torchvision.models.detection
    if initialization == "coco-detector":
        weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        predictor = model.roi_heads.box_predictor
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
            predictor.cls_score.in_features, 2
        )
        metadata = {
            "mode": initialization,
            "detector_weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
            "backbone_weights": None,
        }
    else:
        backbone_weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            if initialization == "imagenet-backbone"
            else None
        )
        model = detection.fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=2,
        )
        metadata = {
            "mode": initialization,
            "detector_weights": None,
            "backbone_weights": (
                "ResNet50_Weights.IMAGENET1K_V2" if backbone_weights is not None else None
            ),
        }
    return model, metadata


def select_device(torch: Any) -> Any:
    """Prefer CUDA, then Apple Metal, and otherwise use the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

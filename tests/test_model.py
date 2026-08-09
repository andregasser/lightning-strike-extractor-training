from __future__ import annotations

from types import SimpleNamespace

import pytest

from lse_train.model import ARCHITECTURE, build_detector, select_device


def fake_torchvision() -> tuple[SimpleNamespace, list[dict[str, object]], SimpleNamespace]:
    calls: list[dict[str, object]] = []
    predictor = SimpleNamespace(cls_score=SimpleNamespace(in_features=256))
    model = SimpleNamespace(roi_heads=SimpleNamespace(box_predictor=predictor))

    def builder(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return model

    detection = SimpleNamespace(
        FasterRCNN_ResNet50_FPN_V2_Weights=SimpleNamespace(COCO_V1="coco-v1"),
        fasterrcnn_resnet50_fpn_v2=builder,
        faster_rcnn=SimpleNamespace(
            FastRCNNPredictor=lambda features, classes: ("predictor", features, classes)
        ),
    )
    torchvision = SimpleNamespace(
        models=SimpleNamespace(
            ResNet50_Weights=SimpleNamespace(IMAGENET1K_V2="imagenet-v2"),
            detection=detection,
        )
    )
    return torchvision, calls, model


@pytest.mark.parametrize(
    ("initialization", "expected"),
    [
        ("random", {"weights": None, "weights_backbone": None, "num_classes": 2}),
        (
            "imagenet-backbone",
            {"weights": None, "weights_backbone": "imagenet-v2", "num_classes": 2},
        ),
    ],
)
def test_builds_resnet_detector_initialization_modes(
    initialization: str, expected: dict[str, object]
) -> None:
    torchvision, calls, _ = fake_torchvision()
    _, metadata = build_detector(torchvision, initialization=initialization)
    assert ARCHITECTURE == "fasterrcnn_resnet50_fpn_v2"
    assert calls == [expected]
    assert metadata["mode"] == initialization


def test_replaces_coco_classifier_for_lightning_schema() -> None:
    torchvision, calls, model = fake_torchvision()
    _, metadata = build_detector(torchvision, initialization="coco-detector")
    assert calls == [{"weights": "coco-v1"}]
    assert model.roi_heads.box_predictor == ("predictor", 256, 2)
    assert metadata == {
        "mode": "coco-detector",
        "detector_weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        "backbone_weights": None,
    }


def test_rejects_unknown_initialization() -> None:
    torchvision, _, _ = fake_torchvision()
    with pytest.raises(ValueError, match="Unsupported model initialization"):
        build_detector(torchvision, initialization="unknown")


def test_prefers_mps_when_cuda_is_unavailable() -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        device=lambda name: name,
    )
    assert select_device(torch) == "mps"

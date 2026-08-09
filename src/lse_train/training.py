from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .model import ARCHITECTURE, Initialization, build_detector, select_device


def _training_imports() -> tuple[Any, Any]:
    try:
        import torch
        import torchvision
    except ImportError as error:
        raise RuntimeError("Training dependencies are missing; run `uv sync --extra train`") from error
    return torch, torchvision


def _loader(torch: Any, dataset: CocoDetectionDataset, *, shuffle: bool) -> Any:
    return torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=shuffle, collate_fn=lambda batch: tuple(zip(*batch))
    )


def _validation_loss(torch: Any, model: Any, loader: Any, device: Any) -> float:
    """Compute detection loss without updating weights."""
    model.train()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            total += float(sum(model(images, targets).values()).detach().cpu())
            batches += 1
    return total / batches if batches else float("inf")


class CocoDetectionDataset:
    def __init__(self, release: Path, split: str, *, augment: bool = False) -> None:
        self.release = release.resolve()
        self.split = split
        self.augment = augment
        document = json.loads(
            (self.release / "annotations" / f"instances_{split}.json").read_text()
        )
        annotations: dict[int, list[dict[str, Any]]] = {}
        for annotation in document["annotations"]:
            annotations.setdefault(annotation["image_id"], []).append(annotation)
        self.rows = [(image, annotations.get(image["id"], [])) for image in document["images"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        torch, _ = _training_imports()
        image, annotations = self.rows[index]
        pixels = cv2.imread(str(self.release / "images" / self.split / image["file_name"]))
        if pixels is None:
            raise RuntimeError(f"Could not decode training image: {image['file_name']}")
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float() / 255
        boxes = [
            [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            for box in (annotation["bbox"] for annotation in annotations)
        ]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor(image["id"]),
        }
        if self.augment:
            tensor, target = _augment_training_sample(torch, tensor, target)
        return tensor, target


def _augment_training_sample(torch: Any, tensor: Any, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Apply conservative detection-safe augmentation to one training sample."""
    _, _, width = tensor.shape
    if float(torch.rand(1)) < 0.5:
        tensor = torch.flip(tensor, dims=(2,))
        boxes = target["boxes"].clone()
        if len(boxes):
            boxes[:, [0, 2]] = width - target["boxes"][:, [2, 0]]
            target = {**target, "boxes": boxes}
    contrast = 1.0 + float(torch.empty(1).uniform_(-0.15, 0.15))
    brightness = float(torch.empty(1).uniform_(-0.08, 0.08))
    tensor = ((tensor - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)
    return tensor, target


def train(
    release: Path,
    output: Path,
    *,
    epochs: int = 10,
    seed: int = 17,
    patience: int = 5,
    min_delta: float = 0.0,
    augment: bool = True,
    initialization: Initialization = "coco-detector",
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if patience < 0:
        raise ValueError("patience cannot be negative")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative")
    manifest = json.loads((release / "manifest.json").read_text())
    if output.exists():
        raise ValueError(f"Refusing to overwrite training output: {output}")
    torch, torchvision = _training_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = CocoDetectionDataset(release, "train", augment=augment)
    validation_dataset = CocoDetectionDataset(release, "validation")
    if not dataset:
        raise ValueError("Training split is empty")
    if not validation_dataset:
        raise ValueError("Validation split is empty; early stopping requires validation data")
    loader = _loader(torch, dataset, shuffle=True)
    validation_loader = _loader(torch, validation_dataset, shuffle=False)
    model, initialization_metadata = build_detector(
        torchvision, initialization=initialization
    )
    device = select_device(torch)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        losses.append(epoch_loss / len(loader))
        validation_loss = _validation_loss(torch, model, validation_loader, device)
        validation_losses.append(validation_loss)
        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    output.mkdir(parents=True)
    checkpoint = output / "checkpoint.pt"
    torch.save(best_state or model.state_dict(), checkpoint)
    report = {
        "schema_version": 1,
        "dataset_release": manifest["release_id"],
        "architecture": ARCHITECTURE,
        "initialization": initialization_metadata,
        "device": str(device),
        "epochs": epochs,
        "epochs_completed": len(losses),
        "seed": seed,
        "augmentation": {
            "enabled": augment,
            "horizontal_flip_probability": 0.5,
            "contrast_range": [0.85, 1.15],
            "brightness_delta": [-0.08, 0.08],
        },
        "early_stopping": {
            "patience": patience,
            "min_delta": min_delta,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        "losses": losses,
        "validation_losses": validation_losses,
        "checkpoint": checkpoint.name,
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

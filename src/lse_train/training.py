from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .model import ARCHITECTURE, Initialization, build_detector, select_device

DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SCHEDULER_FACTOR = 0.3
DEFAULT_SCHEDULER_PATIENCE = 2
DEFAULT_MIN_LEARNING_RATE = 1e-6
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPSILON = 1e-8
ADAMW_WEIGHT_DECAY = 1e-2


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


def _optimizer_learning_rate(optimizer: Any) -> float:
    rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(rates) != 1:
        raise RuntimeError("Training requires one shared learning rate across optimizer groups")
    return rates.pop()


def _finite_metric(value: float, *, name: str, epoch: int) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite at epoch {epoch}: {value}")
    return value


def _step_learning_rate(
    scheduler: Any | None,
    optimizer: Any,
    *,
    epoch: int,
    validation_loss: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    learning_rate = _optimizer_learning_rate(optimizer)
    if scheduler is not None:
        scheduler.step(validation_loss)
    next_learning_rate = _optimizer_learning_rate(optimizer)
    reduced = next_learning_rate < learning_rate
    history = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "validation_loss": validation_loss,
        "next_learning_rate": next_learning_rate,
        "reduced": reduced,
    }
    reduction = None
    if reduced:
        reduction = {
            "epoch": epoch,
            "reason": "validation_loss_plateau",
            "monitored_value": validation_loss,
            "previous_learning_rate": learning_rate,
            "new_learning_rate": next_learning_rate,
        }
    return history, reduction


def _optimizer_configuration(learning_rate: float) -> dict[str, Any]:
    return {
        "type": "AdamW",
        "learning_rate": learning_rate,
        "betas": list(ADAMW_BETAS),
        "epsilon": ADAMW_EPSILON,
        "weight_decay": ADAMW_WEIGHT_DECAY,
        "amsgrad": False,
        "foreach": None,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }


def _scheduler_configuration(
    *,
    enabled: bool,
    factor: float,
    patience: int,
    threshold: float,
    minimum_learning_rate: float,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "type": "ReduceLROnPlateau" if enabled else None,
        "monitor": "validation_loss" if enabled else None,
        "mode": "min" if enabled else None,
        "factor": factor if enabled else None,
        "patience": patience if enabled else None,
        "threshold": threshold if enabled else None,
        "threshold_mode": "abs" if enabled else None,
        "cooldown": 0 if enabled else None,
        "minimum_learning_rate": minimum_learning_rate if enabled else None,
        "epsilon": ADAMW_EPSILON if enabled else None,
    }


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
    learning_rate: float = DEFAULT_LEARNING_RATE,
    scheduler_enabled: bool = True,
    scheduler_factor: float = DEFAULT_SCHEDULER_FACTOR,
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE,
    min_learning_rate: float = DEFAULT_MIN_LEARNING_RATE,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if patience < 0:
        raise ValueError("patience cannot be negative")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if scheduler_enabled:
        if not 0 < scheduler_factor < 1:
            raise ValueError("scheduler_factor must be between zero and one")
        if scheduler_patience < 0:
            raise ValueError("scheduler_patience cannot be negative")
        if min_learning_rate < 0 or min_learning_rate >= learning_rate:
            raise ValueError("min_learning_rate must be non-negative and below learning_rate")
        if patience <= scheduler_patience + 1:
            raise ValueError(
                "Early-stopping patience must exceed scheduler patience by at least two epochs"
            )
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPSILON,
        weight_decay=ADAMW_WEIGHT_DECAY,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=min_delta,
            threshold_mode="abs",
            cooldown=0,
            min_lr=min_learning_rate,
            eps=ADAMW_EPSILON,
        )
        if scheduler_enabled
        else None
    )
    losses: list[float] = []
    validation_losses: list[float] = []
    learning_rate_history: list[dict[str, Any]] = []
    learning_rate_reductions: list[dict[str, Any]] = []
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
            loss_value = _finite_metric(
                float(loss.detach().cpu()), name="training_loss", epoch=epoch
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss_value
        training_loss = epoch_loss / len(loader)
        losses.append(training_loss)
        validation_loss = _finite_metric(
            _validation_loss(torch, model, validation_loader, device),
            name="validation_loss",
            epoch=epoch,
        )
        validation_losses.append(validation_loss)
        history, reduction = _step_learning_rate(
            scheduler,
            optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
        )
        learning_rate_history.append(history)
        if reduction is not None:
            learning_rate_reductions.append(reduction)
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
        "schema_version": 2,
        "dataset_release": manifest["release_id"],
        "dataset_composition": next(
            (
                item
                for item in manifest.get("files", [])
                if item["path"] == "reports/dataset-composition.json"
            ),
            None,
        ),
        "dataset_split_audit": next(
            (
                item
                for item in manifest.get("files", [])
                if item["path"] == "reports/split-audit.json"
            ),
            None,
        ),
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
        "optimizer": _optimizer_configuration(learning_rate),
        "scheduler": _scheduler_configuration(
            enabled=scheduler_enabled,
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=min_delta,
            minimum_learning_rate=min_learning_rate,
        ),
        "learning_rate_history": learning_rate_history,
        "learning_rate_reductions": learning_rate_reductions,
        "final_learning_rate": _optimizer_learning_rate(optimizer),
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

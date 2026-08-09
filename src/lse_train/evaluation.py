from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .model import build_detector
from .training import CocoDetectionDataset, _training_imports

Prediction = tuple[int, tuple[float, float, float, float], float]
GroundTruth = dict[int, list[tuple[float, float, float, float]]]
COCO_IOU_THRESHOLDS = tuple(round(0.5 + step * 0.05, 2) for step in range(10))


def _iou(first: Any, second: Any) -> float:
    x0, y0 = max(float(first[0]), float(second[0])), max(float(first[1]), float(second[1]))
    x1, y1 = min(float(first[2]), float(second[2])), min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _ranked_metrics(
    predictions: list[Prediction], ground_truth: GroundTruth, *, iou_threshold: float
) -> dict[str, Any]:
    """Match ranked predictions once and compute counts, PR points and COCO-style AP."""
    ranked = sorted(predictions, key=lambda prediction: prediction[2], reverse=True)
    matched: dict[int, set[int]] = {image_id: set() for image_id in ground_truth}
    total_expected = sum(len(boxes) for boxes in ground_truth.values())
    true_positives = false_positives = 0
    curve: list[dict[str, float | int]] = []
    for rank, (image_id, box, score) in enumerate(ranked):
        expected = ground_truth.get(image_id, [])
        candidates = [
            (index, _iou(box, truth))
            for index, truth in enumerate(expected)
            if index not in matched.setdefault(image_id, set())
        ]
        best = max(candidates, key=lambda item: item[1], default=None)
        if best is not None and best[1] >= iou_threshold:
            matched[image_id].add(best[0])
            true_positives += 1
        else:
            false_positives += 1
        if rank + 1 < len(ranked) and ranked[rank + 1][2] == score:
            continue
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / total_expected if total_expected else 0.0
        curve.append(
            {
                "score_threshold": score,
                "precision": precision,
                "recall": recall,
                "true_positives": true_positives,
                "false_positives": false_positives,
            }
        )
    false_negatives = total_expected - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = true_positives / total_expected if total_expected else 0.0
    average_precision = None
    if total_expected:
        interpolated = [
            max(
                (float(point["precision"]) for point in curve if point["recall"] >= level),
                default=0.0,
            )
            for level in (step / 100 for step in range(101))
        ]
        average_precision = sum(interpolated) / len(interpolated)
    return {
        "average_precision": average_precision,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "curve": curve,
    }


def _box_values(box: Any) -> tuple[float, float, float, float]:
    values = [float(value) for value in box]
    return values[0], values[1], values[2], values[3]


def _write_curve(path: Path, curve: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "score_threshold",
                "precision",
                "recall",
                "true_positives",
                "false_positives",
            ),
        )
        writer.writeheader()
        writer.writerows(curve)


def evaluate(
    release: Path,
    checkpoint: Path,
    output: Path,
    *,
    split: str = "validation",
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    torch, torchvision = _training_imports()
    dataset = CocoDetectionDataset(release, split)
    if not dataset:
        raise ValueError(f"{split} split is empty")
    model, _ = build_detector(torchvision, initialization="random")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.roi_heads.score_thresh = 0.0
    model.eval()
    predictions: list[Prediction] = []
    ground_truth: GroundTruth = {}
    with torch.no_grad():
        for image_id, (image, target) in enumerate(dataset):
            prediction = model([image])[0]
            ground_truth[image_id] = [_box_values(box) for box in target["boxes"]]
            predictions.extend(
                (image_id, _box_values(box), float(score))
                for box, score, label in zip(
                    prediction["boxes"], prediction["scores"], prediction["labels"]
                )
                if int(label) == 1
            )

    operating = _ranked_metrics(
        [prediction for prediction in predictions if prediction[2] >= score_threshold],
        ground_truth,
        iou_threshold=iou_threshold,
    )
    by_iou = {
        f"{threshold:.2f}": _ranked_metrics(
            predictions, ground_truth, iou_threshold=threshold
        )["average_precision"]
        for threshold in COCO_IOU_THRESHOLDS
    }
    defined_average_precision = [value for value in by_iou.values() if value is not None]
    precision_recall = _ranked_metrics(predictions, ground_truth, iou_threshold=0.5)
    curve_path = output.with_name(f"{output.stem}-pr-curve.csv")
    report = {
        "schema_version": 2,
        "split": split,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": operating["true_positives"],
        "false_positives": operating["false_positives"],
        "false_negatives": operating["false_negatives"],
        "precision": operating["precision"],
        "recall": operating["recall"],
        "ap_50": by_iou["0.50"],
        "map_50": by_iou["0.50"],
        "map_50_95": (
            sum(defined_average_precision) / len(defined_average_precision)
            if defined_average_precision
            else None
        ),
        "average_precision_by_iou": by_iou,
        "maximum_detections_per_image": model.roi_heads.detections_per_img,
        "model_score_floor": model.roi_heads.score_thresh,
        "precision_recall_curve": {
            "iou_threshold": 0.5,
            "points": precision_recall["curve"],
            "csv": curve_path.name,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    _write_curve(curve_path, precision_recall["curve"])
    return report

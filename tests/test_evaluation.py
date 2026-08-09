from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from lse_train.evaluation import _iou, _ranked_metrics, _write_curve, evaluate


def test_iou_measures_box_overlap() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert _iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_perfect_ranked_predictions_have_full_average_precision() -> None:
    ground_truth = {0: [(0, 0, 10, 10)], 1: [(5, 5, 15, 15)]}
    predictions = [
        (0, (0, 0, 10, 10), 0.9),
        (1, (5, 5, 15, 15), 0.8),
    ]
    metrics = _ranked_metrics(predictions, ground_truth, iou_threshold=0.5)
    assert metrics["average_precision"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_duplicate_detections_count_as_false_positives() -> None:
    metrics = _ranked_metrics(
        [
            (0, (0, 0, 10, 10), 0.9),
            (0, (0, 0, 10, 10), 0.8),
            (0, (20, 20, 30, 30), 0.7),
        ],
        {0: [(0, 0, 10, 10)]},
        iou_threshold=0.5,
    )
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 0
    assert metrics["average_precision"] == 1.0


def test_high_scoring_false_positive_reduces_average_precision() -> None:
    metrics = _ranked_metrics(
        [
            (0, (20, 20, 30, 30), 0.9),
            (0, (0, 0, 10, 10), 0.8),
        ],
        {0: [(0, 0, 10, 10)]},
        iou_threshold=0.5,
    )
    assert metrics["average_precision"] == pytest.approx(0.5)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_equal_scores_produce_one_reproducible_curve_point() -> None:
    metrics = _ranked_metrics(
        [
            (0, (0, 0, 10, 10), 0.5),
            (0, (20, 20, 30, 30), 0.5),
        ],
        {0: [(0, 0, 10, 10)]},
        iou_threshold=0.5,
    )
    assert metrics["curve"] == [
        {
            "score_threshold": 0.5,
            "precision": 0.5,
            "recall": 1.0,
            "true_positives": 1,
            "false_positives": 1,
        }
    ]


def test_missing_ground_truth_marks_average_precision_undefined() -> None:
    metrics = _ranked_metrics(
        [(0, (0, 0, 10, 10), 0.9)],
        {0: []},
        iou_threshold=0.5,
    )
    assert metrics["average_precision"] is None
    assert metrics["false_positives"] == 1
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_writes_precision_recall_curve_csv() -> None:
    curve = [
        {
            "score_threshold": 0.9,
            "precision": 1.0,
            "recall": 0.5,
            "true_positives": 1,
            "false_positives": 0,
        }
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "curve.csv"
        _write_curve(path, curve)
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == [
        {
            "score_threshold": "0.9",
            "precision": "1.0",
            "recall": "0.5",
            "true_positives": "1",
            "false_positives": "0",
        }
    ]


@pytest.mark.parametrize(
    ("score_threshold", "iou_threshold", "message"),
    [
        (-0.1, 0.5, "score_threshold"),
        (1.1, 0.5, "score_threshold"),
        (0.25, 0.0, "iou_threshold"),
        (0.25, 1.1, "iou_threshold"),
    ],
)
def test_rejects_invalid_evaluation_thresholds(
    score_threshold: float, iou_threshold: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate(
            Path("missing-release"),
            Path("missing-checkpoint"),
            Path("missing-output"),
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )

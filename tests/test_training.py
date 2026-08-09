from __future__ import annotations

from pathlib import Path

import pytest

from lse_train import cli
from lse_train.training import (
    _finite_metric,
    _optimizer_configuration,
    _scheduler_configuration,
    _step_learning_rate,
    train,
)


class FakeOptimizer:
    def __init__(self, learning_rate: float) -> None:
        self.param_groups = [{"lr": learning_rate}]


class ReducingScheduler:
    def __init__(self, optimizer: FakeOptimizer) -> None:
        self.optimizer = optimizer

    def step(self, validation_loss: float) -> None:
        assert validation_loss == 0.75
        self.optimizer.param_groups[0]["lr"] = 3e-5


def test_records_learning_rate_reduction_after_validation() -> None:
    optimizer = FakeOptimizer(1e-4)

    history, reduction = _step_learning_rate(
        ReducingScheduler(optimizer),
        optimizer,
        epoch=4,
        validation_loss=0.75,
    )

    assert history == {
        "epoch": 4,
        "learning_rate": 1e-4,
        "validation_loss": 0.75,
        "next_learning_rate": 3e-5,
        "reduced": True,
    }
    assert reduction == {
        "epoch": 4,
        "reason": "validation_loss_plateau",
        "monitored_value": 0.75,
        "previous_learning_rate": 1e-4,
        "new_learning_rate": 3e-5,
    }


def test_records_fixed_learning_rate_without_scheduler() -> None:
    optimizer = FakeOptimizer(1e-4)

    history, reduction = _step_learning_rate(
        None,
        optimizer,
        epoch=1,
        validation_loss=1.25,
    )

    assert history["learning_rate"] == 1e-4
    assert history["next_learning_rate"] == 1e-4
    assert history["reduced"] is False
    assert reduction is None


def test_records_complete_optimizer_and_scheduler_configuration() -> None:
    assert _optimizer_configuration(1e-4) == {
        "type": "AdamW",
        "learning_rate": 1e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "foreach": None,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    assert _scheduler_configuration(
        enabled=True,
        factor=0.3,
        patience=2,
        threshold=0.01,
        minimum_learning_rate=1e-6,
    ) == {
        "enabled": True,
        "type": "ReduceLROnPlateau",
        "monitor": "validation_loss",
        "mode": "min",
        "factor": 0.3,
        "patience": 2,
        "threshold": 0.01,
        "threshold_mode": "abs",
        "cooldown": 0,
        "minimum_learning_rate": 1e-6,
        "epsilon": 1e-8,
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"scheduler_factor": 1.0}, "scheduler_factor must be between zero and one"),
        ({"scheduler_patience": -1}, "scheduler_patience cannot be negative"),
        (
            {"min_learning_rate": 1e-4},
            "min_learning_rate must be non-negative and below learning_rate",
        ),
        (
            {"patience": 3, "scheduler_patience": 2},
            "Early-stopping patience must exceed scheduler patience by at least two epochs",
        ),
    ],
)
def test_rejects_invalid_scheduler_configuration(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        train(Path("missing-release"), Path("missing-output"), **options)


def test_fixed_rate_allows_short_early_stopping_patience() -> None:
    with pytest.raises(FileNotFoundError):
        train(
            Path("missing-release"),
            Path("missing-output"),
            patience=0,
            scheduler_enabled=False,
            scheduler_factor=2.0,
            scheduler_patience=-1,
            min_learning_rate=1e-4,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_training_metrics(value: float) -> None:
    with pytest.raises(RuntimeError, match="validation_loss is not finite at epoch 3"):
        _finite_metric(value, name="validation_loss", epoch=3)


def test_cli_passes_scheduler_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_train(release: Path, output: Path, **options: object) -> dict[str, object]:
        captured.update({"release": release, "output": output, **options})
        return {"status": "ok"}

    monkeypatch.setattr(cli, "train", fake_train)

    assert (
        cli.main(
            [
                "train",
                "release",
                "--output",
                "experiment",
                "--learning-rate",
                "0.0002",
                "--scheduler-factor",
                "0.5",
                "--scheduler-patience",
                "3",
                "--min-learning-rate",
                "0.000002",
            ]
        )
        == 0
    )
    assert captured["learning_rate"] == 2e-4
    assert captured["scheduler_enabled"] is True
    assert captured["scheduler_factor"] == 0.5
    assert captured["scheduler_patience"] == 3
    assert captured["min_learning_rate"] == 2e-6

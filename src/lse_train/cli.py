from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate
from .handoff import import_handoff
from .model import INITIALIZATIONS
from .onnx_release import export_onnx_release
from .release import build_release
from .training import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_LEARNING_RATE,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_SCHEDULER_PATIENCE,
    train,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lse-train")
    commands = parser.add_subparsers(dest="command", required=True)
    release = commands.add_parser("release", help="Build an immutable verified dataset release")
    release.add_argument("campaigns", nargs="+", type=Path)
    release.add_argument("--output", required=True, type=Path)
    release.add_argument("--release-id", required=True)
    handoff = commands.add_parser("import-handoff", help="Import a CLI frame handoff for annotation")
    handoff.add_argument("handoff", type=Path)
    handoff.add_argument("--output", required=True, type=Path)
    handoff.add_argument("--image-base-url", default="http://localhost:8001/images")
    training = commands.add_parser("train", help="Train the closed-set detector")
    training.add_argument("release", type=Path)
    training.add_argument("--output", required=True, type=Path)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--min-delta", type=float, default=0.0)
    training.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Set the initial AdamW learning rate",
    )
    training.add_argument(
        "--scheduler-factor",
        type=float,
        default=DEFAULT_SCHEDULER_FACTOR,
        help="Multiply the learning rate by this factor after a validation plateau",
    )
    training.add_argument(
        "--scheduler-patience",
        type=int,
        default=DEFAULT_SCHEDULER_PATIENCE,
        help="Wait this many plateau epochs before reducing the learning rate",
    )
    training.add_argument(
        "--min-learning-rate",
        type=float,
        default=DEFAULT_MIN_LEARNING_RATE,
        help="Do not reduce the learning rate below this value",
    )
    training.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Keep a fixed learning rate for a controlled baseline run",
    )
    training.add_argument(
        "--initialization",
        choices=INITIALIZATIONS,
        default="coco-detector",
        help="Initialize from COCO detector weights, ImageNet backbone weights, or randomly",
    )
    training.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable training image augmentation for a strict baseline run",
    )
    evaluation = commands.add_parser("evaluate", help="Evaluate one trained checkpoint")
    evaluation.add_argument("release", type=Path)
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluation.add_argument("--score-threshold", type=float, default=0.25)
    evaluation.add_argument("--iou-threshold", type=float, default=0.5)
    export = commands.add_parser("export-onnx", help="Export and parity-check an ONNX release")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("--training-report", required=True, type=Path)
    export.add_argument("--evaluation-report", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    if args.command == "import-handoff":
        result = import_handoff(
            args.handoff,
            args.output,
            image_base_url=args.image_base_url,
        )
    elif args.command == "release":
        result = build_release(args.campaigns, args.output, release_id=args.release_id)
    elif args.command == "train":
        result = train(
            args.release,
            args.output,
            epochs=args.epochs,
            seed=args.seed,
            patience=args.patience,
            min_delta=args.min_delta,
            augment=not args.no_augmentation,
            initialization=args.initialization,
            learning_rate=args.learning_rate,
            scheduler_enabled=not args.no_scheduler,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            min_learning_rate=args.min_learning_rate,
        )
    elif args.command == "evaluate":
        result = evaluate(
            args.release,
            args.checkpoint,
            args.output,
            split=args.split,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
    else:
        result = export_onnx_release(
            args.checkpoint,
            args.training_report,
            args.evaluation_report,
            args.output,
            version=args.version,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

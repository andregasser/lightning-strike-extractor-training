from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from lse_train.coco import CATEGORY
from lse_train.release import build_release, sha256


def _campaign(root: Path) -> Path:
    campaign = root / "campaign"
    image_specs = {
        "train": [
            {
                "name": "positive.jpg",
                "source_id": "storm-a",
                "camera": "sony-a7",
                "recording_conditions": ["night", "rain"],
                "rare_cases": ["frame-edge"],
                "boxes": [([1, 2, 8, 10], ["faint-channel"]), ([12, 3, 4, 8], [])],
            },
            {"name": "negative.jpg", "source_id": "storm-a", "boxes": []},
        ],
        "validation": [
            {
                "name": "validation.jpg",
                "source_id": "storm-b",
                "camera": "iphone-15",
                "recording_conditions": ["day"],
                "boxes": [([2, 2, 5, 6], [])],
            }
        ],
        "test": [],
    }
    for split, specs in image_specs.items():
        image_dir = campaign / "images" / split
        image_dir.mkdir(parents=True)
        images = []
        annotations = []
        for image_id, spec in enumerate(specs, 1):
            pixels = np.full((20, 30, 3), image_id + len(split), dtype=np.uint8)
            cv2.imwrite(str(image_dir / spec["name"]), pixels)
            image = {
                "id": image_id,
                "file_name": spec["name"],
                "width": 30,
                "height": 20,
                "source_id": spec["source_id"],
            }
            for field in ("camera", "recording_conditions", "rare_cases"):
                if field in spec:
                    image[field] = spec[field]
            images.append(image)
            for box, rare_cases in spec["boxes"]:
                attributes = {"verified": True}
                if rare_cases:
                    attributes["rare_cases"] = rare_cases
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": box,
                        "attributes": attributes,
                    }
                )
        annotation_path = campaign / "annotations" / f"instances_{split}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": [CATEGORY]})
        )
    (campaign / "manifest.json").write_text(json.dumps({"schema_version": 1}))
    return campaign


def test_release_writes_json_and_markdown_composition_reports() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "release"
        manifest = build_release([_campaign(root)], output, release_id="lightning-v1")

        report_path = output / "reports" / "dataset-composition.json"
        markdown_path = output / "reports" / "dataset-composition.md"
        report = json.loads(report_path.read_text())

        assert report["totals"] == {
            "images": 3,
            "positive_images": 2,
            "negative_images": 1,
            "boxes": 3,
        }
        assert report["boxes_per_image"] == {"0": 1, "1": 1, "2": 1}
        assert report["distributions"]["sources"] == {"storm-a": 2, "storm-b": 1}
        assert report["distributions"]["cameras"] == {
            "iphone-15": 1,
            "sony-a7": 1,
            "unknown": 1,
        }
        assert report["distributions"]["recording_conditions"] == {
            "day": 1,
            "night": 1,
            "rain": 1,
            "unknown": 1,
        }
        assert report["distributions"]["rare_cases"] == {
            "faint-channel": 1,
            "frame-edge": 1,
        }
        assert "| train | 2 | 1 | 1 | 2 | 1 |" in markdown_path.read_text()
        tracked = {item["path"]: item["sha256"] for item in manifest["files"]}
        assert tracked["reports/dataset-composition.json"] == sha256(report_path)
        assert tracked["reports/dataset-composition.md"] == sha256(markdown_path)


def test_release_reports_missing_metadata() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "release"
        build_release([_campaign(root)], output, release_id="lightning-v1")
        report = json.loads((output / "reports" / "dataset-composition.json").read_text())
        warning_codes = {warning["code"] for warning in report["warnings"]}

        assert "missing-camera" in warning_codes
        assert "missing-recording-conditions" in warning_codes
        assert report["warning_thresholds"] == {
            "minimum_positive_image_fraction": 0.2,
            "maximum_positive_image_fraction": 0.8,
        }


def test_release_preserves_composition_metadata() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "release"
        build_release([_campaign(root)], output, release_id="lightning-v1")
        document = json.loads(
            (output / "annotations" / "instances_train.json").read_text()
        )
        positive = next(image for image in document["images"] if image.get("camera"))
        rare_annotation = next(
            annotation
            for annotation in document["annotations"]
            if annotation["attributes"].get("rare_cases")
        )

        assert positive["recording_conditions"] == ["night", "rain"]
        assert positive["rare_cases"] == ["frame-edge"]
        assert rare_annotation["attributes"]["rare_cases"] == ["faint-channel"]

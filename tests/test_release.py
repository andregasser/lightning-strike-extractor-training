from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lse_train.release import build_release, sha256


def campaign(root: Path, name: str, *, box: list[int] | None = None) -> Path:
    path = root / name
    for split in ("train", "validation", "test"):
        image_dir = path / "images" / split
        image_dir.mkdir(parents=True)
        images = []
        annotations = []
        if split == "train":
            cv2.imwrite(str(image_dir / "frame.jpg"), np.zeros((20, 30, 3), dtype=np.uint8))
            images.append({"id": 1, "file_name": "frame.jpg", "width": 30, "height": 20, "source_id": "source-a"})
            if box is not None:
                annotations.append({"id": 1, "image_id": 1, "category_id": 1, "bbox": box, "attributes": {"verified": True}})
        document = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "lightning_channel", "supercategory": "weather"}]}
        annotation_path = path / "annotations" / f"instances_{split}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(json.dumps(document))
    (path / "manifest.json").write_text(json.dumps({"campaign": name}))
    return path


def test_builds_hashed_immutable_release() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = campaign(root, "campaign-a", box=[1, 2, 10, 12])
        output = root / "release"
        manifest = build_release([source], output, release_id="lightning-v1")
        image = next((output / "images" / "train").iterdir())
        assert image.stem == sha256(image)
        assert manifest["splits"]["train"] == {"images": 1, "annotations": 1, "sources": 1}
        assert json.loads((output / "manifest.json").read_text())["release_id"] == "lightning-v1"
        assert (output / "reports" / "dataset-composition.json").is_file()
        assert (output / "reports" / "dataset-composition.md").is_file()
        composition = json.loads(
            (output / "reports" / "dataset-composition.json").read_text()
        )
        assert "class-imbalance" in {
            warning["code"] for warning in composition["warnings"]
        }


def test_rejects_conflicting_annotations_for_identical_image() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = campaign(root, "campaign-a", box=[1, 2, 10, 12])
        second = campaign(root, "campaign-b", box=[2, 2, 10, 12])
        with pytest.raises(ValueError, match="Conflicting annotations"):
            build_release([first, second], root / "release", release_id="lightning-v1")


def test_rejects_invalid_composition_metadata_before_publication() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = campaign(root, "campaign-a")
        annotation_path = source / "annotations" / "instances_train.json"
        document = json.loads(annotation_path.read_text())
        document["images"][0]["recording_conditions"] = ["night", "night"]
        annotation_path.write_text(json.dumps(document))
        output = root / "release"

        with pytest.raises(ValueError, match="invalid recording_conditions metadata"):
            build_release([source], output, release_id="lightning-v1")

        assert not output.exists()

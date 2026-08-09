from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lse_train.handoff import import_handoff


def make_handoff(root: Path) -> Path:
    handoff = root / "handoff"
    image_dir = handoff / "images" / "source-a"
    image_dir.mkdir(parents=True)
    image = image_dir / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((20, 30, 3), dtype=np.uint8))
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    relative = image.relative_to(handoff).as_posix()
    (handoff / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frames": [
                    {
                        "file_name": relative,
                        "sha256": digest,
                        "source_id": "source-a",
                        "frame_number": 4,
                        "time": 0.4,
                    }
                ],
            }
        )
    )
    return handoff


def test_imports_handoff_and_preserves_provenance() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        campaign = import_handoff(make_handoff(root), root / "campaign")
        assert campaign["tasks"] == 1
        task = json.loads((root / "campaign/annotation/tasks.json").read_text())[0]
        assert task["data"]["source_id"] == "source-a"
        assert task["data"]["frame_provenance"]["frame_number"] == 4
        assert (root / "campaign/serve/images/source-a__frame.jpg").is_file()


def test_rejects_checksum_mismatch_without_publishing_output() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        handoff = make_handoff(root)
        manifest = json.loads((handoff / "manifest.json").read_text())
        manifest["frames"][0]["sha256"] = "0" * 64
        (handoff / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="checksum mismatch"):
            import_handoff(handoff, root / "campaign")
        assert not (root / "campaign").exists()

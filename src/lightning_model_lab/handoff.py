from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2

LABEL_CONFIG = """<View>
  <Header value="Draw a tight box around every visible lightning channel."/>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="lightning_channel" background="#FFD200"/>
  </RectangleLabels>
</View>
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Handoff file_name must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Handoff file_name is unsafe: {value}")
    return relative


def _load_handoff(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Handoff manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported frame handoff schema; expected schema_version 1")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Frame handoff contains no frames")
    return manifest, frames


def import_handoff(
    handoff: Path,
    output: Path,
    *,
    image_base_url: str = "http://localhost:8001/images",
) -> dict[str, Any]:
    """Validate a CLI frame handoff and publish an annotation campaign."""
    handoff = handoff.resolve()
    output = output.resolve()
    if not handoff.is_dir():
        raise ValueError(f"Frame handoff does not exist: {handoff}")
    if output.exists():
        raise ValueError(f"Refusing to overwrite annotation campaign: {output}")
    if not image_base_url.startswith(("http://", "https://")):
        raise ValueError("image_base_url must be an HTTP or HTTPS URL")
    manifest, frames = _load_handoff(handoff)
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_names: set[str] = set()
    for index, frame in enumerate(frames, 1):
        if not isinstance(frame, dict):
            raise TypeError(f"Handoff frame {index} must be an object")
        relative = _safe_relative(frame.get("file_name"))
        source_id = frame.get("source_id")
        expected_hash = frame.get("sha256")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"Handoff frame {index} has no source_id")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Handoff frame {index} has no valid sha256")
        source = handoff / relative
        if not source.is_file():
            raise ValueError(f"Handoff frame does not exist: {source}")
        actual_hash = _sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(f"Handoff checksum mismatch: {relative}")
        if actual_hash in seen_hashes:
            raise ValueError(f"Handoff contains duplicate image hash: {actual_hash}")
        destination_name = f"{source_id}__{relative.name}"
        if destination_name in seen_names:
            raise ValueError(f"Handoff produces duplicate annotation filename: {destination_name}")
        pixels = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f"Handoff image cannot be decoded: {source}")
        height, width = pixels.shape[:2]
        seen_hashes.add(actual_hash)
        seen_names.add(destination_name)
        records.append(
            {
                "source": source,
                "source_id": source_id,
                "original_file_name": relative.as_posix(),
                "served_name": destination_name,
                "sha256": actual_hash,
                "width": width,
                "height": height,
                "frame": frame,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        image_root = staged / "serve" / "images"
        image_root.mkdir(parents=True)
        tasks: list[dict[str, Any]] = []
        by_source: dict[str, int] = defaultdict(int)
        for task_id, record in enumerate(records, 1):
            shutil.copy2(record["source"], image_root / record["served_name"])
            by_source[record["source_id"]] += 1
            tasks.append(
                {
                    "id": task_id,
                    "data": {
                        "image": f"{image_base_url.rstrip('/')}/{quote(record['served_name'])}",
                        "source_id": record["source_id"],
                        "original_file_name": record["original_file_name"],
                        "frame_provenance": record["frame"],
                    },
                }
            )
        task_path = staged / "annotation" / "tasks.json"
        task_path.parent.mkdir(parents=True)
        task_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
        (staged / "annotation" / "label-config.xml").write_text(LABEL_CONFIG)
        campaign_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "source_format": "lightning frame handoff",
            "source_handoff": str(handoff),
            "source_handoff_manifest": manifest,
            "image_base_url": image_base_url,
            "tasks": len(tasks),
            "sources": dict(sorted(by_source.items())),
            "files": [
                {
                    "file_name": record["served_name"],
                    "source_id": record["source_id"],
                    "sha256": record["sha256"],
                    "width": record["width"],
                    "height": record["height"],
                }
                for record in records
            ],
        }
        (staged / "manifest.json").write_text(
            json.dumps(campaign_manifest, indent=2, ensure_ascii=False) + "\n"
        )
        os.replace(staged, output)
    return campaign_manifest

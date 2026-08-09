from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

SPLITS = ("train", "validation", "test")
GROUP_FIELDS = ("source_id", "recording_group_id", "event_group_id", "duplicate_group_id")
OPTIONAL_GROUP_FIELDS = GROUP_FIELDS[1:]
DHASH_DISTANCE_THRESHOLD = 4
_BAND_WIDTHS = (13, 13, 13, 13, 12)


def _difference_hash(path: Path) -> int:
    pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError(f"Release image cannot be decoded for split audit: {path}")
    resized = cv2.resize(pixels, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def _bands(value: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    shift = 64
    for index, width in enumerate(_BAND_WIDTHS):
        shift -= width
        bands.append((index, (value >> shift) & ((1 << width) - 1)))
    return bands


def _near_duplicate_components(images_by_hash: dict[int, list[dict[str, Any]]]) -> list[list[int]]:
    hashes = sorted(images_by_hash)
    parent = {value: value for value in hashes}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for value in hashes:
        candidates: set[int] = set()
        for band in _bands(value):
            candidates.update(buckets[band])
        for candidate in candidates:
            if (value ^ candidate).bit_count() <= DHASH_DISTANCE_THRESHOLD:
                union(value, candidate)
        for band in _bands(value):
            buckets[band].append(value)

    components: dict[int, list[int]] = defaultdict(list)
    for value in hashes:
        components[find(value)].append(value)
    return list(components.values())


def build_split_audit(release: Path, *, release_id: str) -> dict[str, Any]:
    """Audit hard split groups and visual near-duplicates in a staged release."""
    group_assignments: dict[str, dict[str, str]] = {field: {} for field in GROUP_FIELDS}
    missing_metadata = {field: 0 for field in OPTIONAL_GROUP_FIELDS}
    images_by_hash: dict[int, list[dict[str, Any]]] = defaultdict(list)
    image_count = 0

    for split in SPLITS:
        document = json.loads(
            (release / "annotations" / f"instances_{split}.json").read_text()
        )
        for image in document["images"]:
            image_count += 1
            for field in GROUP_FIELDS:
                group_id = image.get(field)
                if group_id is None:
                    if field in missing_metadata:
                        missing_metadata[field] += 1
                    continue
                previous = group_assignments[field].setdefault(group_id, split)
                if previous != split:
                    raise ValueError(
                        f"{field} {group_id} appears in both {previous} and {split}"
                    )
            path = release / "images" / split / image["file_name"]
            difference_hash = _difference_hash(path)
            images_by_hash[difference_hash].append(
                {
                    "path": path.relative_to(release).as_posix(),
                    "split": split,
                    "source_id": image["source_id"],
                    "dhash": f"{difference_hash:016x}",
                }
            )

    near_duplicate_groups: list[dict[str, Any]] = []
    for component in _near_duplicate_components(images_by_hash):
        images = sorted(
            (image for value in component for image in images_by_hash[value]),
            key=lambda item: item["path"],
        )
        splits = sorted({image["split"] for image in images})
        if len(splits) > 1:
            near_duplicate_groups.append(
                {
                    "splits": splits,
                    "images": images,
                    "link_hamming_distance_threshold": DHASH_DISTANCE_THRESHOLD,
                }
            )
    near_duplicate_groups.sort(key=lambda item: item["images"][0]["path"])

    warnings = []
    if near_duplicate_groups:
        warnings.append(
            {
                "code": "cross-split-near-duplicates",
                "message": (
                    f"{len(near_duplicate_groups)} visual similarity groups cross split boundaries "
                    "and require review."
                ),
            }
        )
    for field, missing in missing_metadata.items():
        if missing:
            warnings.append(
                {
                    "code": f"missing-{field.replace('_', '-')}",
                    "message": f"{missing} images have no {field} metadata.",
                }
            )

    return {
        "schema_version": 1,
        "dataset_release": release_id,
        "status": "review-required" if near_duplicate_groups else "passed",
        "images_checked": image_count,
        "hard_group_checks": {
            field: {
                "status": "passed",
                "groups": len(assignments),
                "assignments": dict(sorted(assignments.items())),
            }
            for field, assignments in group_assignments.items()
        },
        "near_duplicate_check": {
            "algorithm": "difference-hash-64",
            "maximum_hamming_distance": DHASH_DISTANCE_THRESHOLD,
            "unique_hashes": len(images_by_hash),
            "cross_split_groups": near_duplicate_groups,
        },
        "missing_group_metadata": missing_metadata,
        "warnings": warnings,
    }


def render_split_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Split audit: {report['dataset_release']}",
        "",
        f"- Status: {report['status']}",
        f"- Images checked: {report['images_checked']}",
        "- Near-duplicate algorithm: difference-hash-64",
        (
            "- Maximum near-duplicate Hamming distance: "
            f"{report['near_duplicate_check']['maximum_hamming_distance']}"
        ),
        "",
        "## Hard groups",
        "",
    ]
    for field, check in report["hard_group_checks"].items():
        lines.append(f"- {field}: {check['groups']} groups, {check['status']}")
    lines.extend(["", "## Cross-split near-duplicate groups", ""])
    groups = report["near_duplicate_check"]["cross_split_groups"]
    if groups:
        for index, group in enumerate(groups, 1):
            lines.append(f"### Group {index}")
            lines.append("")
            lines.append(f"Splits: {', '.join(group['splits'])}")
            lines.append("")
            lines.extend(
                f"- {image['split']}: `{image['path']}` ({image['dhash']})"
                for image in group["images"]
            )
            lines.append("")
    else:
        lines.extend(["No cross-split near-duplicates detected.", ""])
    lines.extend(["## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- [{item['code']}] {item['message']}" for item in report["warnings"])
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"

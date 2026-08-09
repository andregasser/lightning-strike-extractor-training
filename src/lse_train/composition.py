from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

UNKNOWN = "unknown"
SPLITS = ("train", "validation", "test")
MIN_POSITIVE_FRACTION = 0.2
MAX_POSITIVE_FRACTION = 0.8


def _distribution(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _metadata_values(image: dict[str, Any], field: str) -> list[str]:
    value = image.get(field)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return value
    return []


def build_composition_report(release: Path, *, release_id: str) -> dict[str, Any]:
    """Summarize the composition of a verified dataset release."""
    totals = {"images": 0, "positive_images": 0, "negative_images": 0, "boxes": 0}
    splits: dict[str, dict[str, Any]] = {}
    all_sources: list[str] = []
    all_cameras: list[str] = []
    all_conditions: list[str] = []
    all_rare_cases: list[str] = []
    boxes_per_image: Counter[int] = Counter()

    for split in SPLITS:
        annotation_path = release / "annotations" / f"instances_{split}.json"
        document = json.loads(annotation_path.read_text())
        annotations_by_image: dict[int, int] = defaultdict(int)
        rare_cases_by_image: dict[int, list[str]] = defaultdict(list)
        for annotation in document["annotations"]:
            annotations_by_image[annotation["image_id"]] += 1
            rare_cases_by_image[annotation["image_id"]].extend(
                _metadata_values(annotation.get("attributes", {}), "rare_cases")
            )

        sources: list[str] = []
        cameras: list[str] = []
        conditions: list[str] = []
        rare_cases: list[str] = []
        positive_images = 0
        for image in document["images"]:
            box_count = annotations_by_image[image["id"]]
            boxes_per_image[box_count] += 1
            positive_images += box_count > 0
            sources.append(image["source_id"])
            cameras.append(image.get("camera", UNKNOWN))
            image_conditions = _metadata_values(image, "recording_conditions")
            conditions.extend(image_conditions or [UNKNOWN])
            image_rare_cases = _metadata_values(image, "rare_cases")
            image_rare_cases.extend(rare_cases_by_image[image["id"]])
            rare_cases.extend(image_rare_cases)

        image_count = len(document["images"])
        box_count = len(document["annotations"])
        split_report = {
            "images": image_count,
            "positive_images": positive_images,
            "negative_images": image_count - positive_images,
            "boxes": box_count,
            "sources": _distribution(sources),
            "cameras": _distribution(cameras),
            "recording_conditions": _distribution(conditions),
            "rare_cases": _distribution(rare_cases),
        }
        splits[split] = split_report
        totals["images"] += image_count
        totals["positive_images"] += positive_images
        totals["negative_images"] += image_count - positive_images
        totals["boxes"] += box_count
        all_sources.extend(sources)
        all_cameras.extend(cameras)
        all_conditions.extend(conditions)
        all_rare_cases.extend(rare_cases)

    warnings: list[dict[str, str]] = []
    if totals["images"]:
        positive_fraction = totals["positive_images"] / totals["images"]
        if positive_fraction < MIN_POSITIVE_FRACTION or positive_fraction > MAX_POSITIVE_FRACTION:
            warnings.append(
                {
                    "code": "class-imbalance",
                    "message": f"Positive images account for {positive_fraction:.1%} of the dataset.",
                }
            )
    for field, values in (
        ("camera", all_cameras),
        ("recording conditions", all_conditions),
    ):
        unknown = values.count(UNKNOWN)
        if unknown:
            warnings.append(
                {
                    "code": f"missing-{field.replace(' ', '-')}",
                    "message": f"{unknown} images have unknown {field} metadata.",
                }
            )
    if not all_rare_cases:
        warnings.append(
            {"code": "no-rare-cases", "message": "No images or boxes are tagged as rare cases."}
        )

    return {
        "schema_version": 1,
        "dataset_release": release_id,
        "warning_thresholds": {
            "minimum_positive_image_fraction": MIN_POSITIVE_FRACTION,
            "maximum_positive_image_fraction": MAX_POSITIVE_FRACTION,
        },
        "totals": totals,
        "boxes_per_image": {str(count): images for count, images in sorted(boxes_per_image.items())},
        "distributions": {
            "sources": _distribution(all_sources),
            "cameras": _distribution(all_cameras),
            "recording_conditions": _distribution(all_conditions),
            "rare_cases": _distribution(all_rare_cases),
        },
        "splits": splits,
        "warnings": warnings,
    }


def render_composition_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable companion to the canonical JSON report."""
    totals = report["totals"]
    lines = [
        f"# Dataset composition: {report['dataset_release']}",
        "",
        "## Overview",
        "",
        f"- Images: {totals['images']}",
        f"- Positive images: {totals['positive_images']}",
        f"- Negative images: {totals['negative_images']}",
        f"- Bounding boxes: {totals['boxes']}",
        "",
        "## Splits",
        "",
        "| Split | Images | Positive | Negative | Boxes | Sources |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        item = report["splits"][split]
        lines.append(
            f"| {split} | {item['images']} | {item['positive_images']} | "
            f"{item['negative_images']} | {item['boxes']} | {len(item['sources'])} |"
        )
    lines.extend(["", "## Distributions", ""])
    for name, distribution in report["distributions"].items():
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        if distribution:
            lines.extend(f"- {label}: {count}" for label, count in distribution.items())
        else:
            lines.append("- None recorded")
        lines.append("")
    lines.extend(["## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- [{item['code']}] {item['message']}" for item in report["warnings"])
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"

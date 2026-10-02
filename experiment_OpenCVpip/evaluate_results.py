#!/usr/bin/env python3
"""Score OpenCV-pip JSON outputs against the experiment ground truth."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path, help="Result directories")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("experiment_OpenCVpip/ground_truth.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def _tile_key(values: list[int] | tuple[int, int]) -> tuple[int, int]:
    a, b = int(values[0]), int(values[1])
    return (min(a, b), max(a, b))


def _load_results(result_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for directory in result_dirs:
        if not directory.is_dir():
            raise ValueError(f"Result directory does not exist: {directory}")
        for path in sorted(directory.glob("*.opencv-pip.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            image_name = Path(payload["input_image"]).name
            if image_name in loaded:
                raise ValueError(f"Duplicate result for {image_name}")
            loaded[image_name] = payload
    return loaded


def _score_image(name: str, truth: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected = Counter(_tile_key(tile) for tile in truth["tiles"])
    candidates = list(result.get("dominoes", []))
    readable_candidates = [item for item in candidates if bool(item.get("readable"))]
    predicted = Counter(
        _tile_key((item["end_0_pips"], item["end_1_pips"]))
        for item in readable_candidates
    )
    recovered = sum((expected & predicted).values())
    all_recovered = recovered == sum(expected.values())
    exact = all_recovered and predicted == expected and len(candidates) == len(readable_candidates)
    posture_correct = sum(
        item.get("posture") == truth["posture"] for item in candidates
    )
    return {
        "image": name,
        "expected": [f"{a}-{b}" for a, b in expected.elements()],
        "predicted": [f"{a}-{b}" for a, b in predicted.elements()],
        "candidates": len(candidates),
        "readable": len(readable_candidates),
        "recovered": recovered,
        "real_tiles": sum(expected.values()),
        "all_recovered_allowing_extras": all_recovered,
        "exact_output_set": exact,
        "posture_correct_candidates": posture_correct,
    }


def evaluate(ground_truth: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_image = []
    missing = []
    for name, truth in ground_truth["images"].items():
        result = results.get(name)
        if result is None:
            missing.append(name)
            continue
        per_image.append(_score_image(name, truth, result))

    totals = {
        "images_scored": len(per_image),
        "images_missing": len(missing),
        "real_tiles": sum(item["real_tiles"] for item in per_image),
        "detector_candidates": sum(item["candidates"] for item in per_image),
        "readable_candidates": sum(item["readable"] for item in per_image),
        "correct_tiles_recovered": sum(item["recovered"] for item in per_image),
        "images_all_recovered_allowing_extras": sum(
            item["all_recovered_allowing_extras"] for item in per_image
        ),
        "images_exact_output_set": sum(item["exact_output_set"] for item in per_image),
        "posture_correct_candidates": sum(
            item["posture_correct_candidates"] for item in per_image
        ),
    }
    return {"totals": totals, "missing_images": missing, "images": per_image}


def _percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def render_report(score: dict[str, Any]) -> str:
    totals = score["totals"]
    lines = [
        "# OpenCV pip-counting benchmark",
        "",
        "Orientation within a tile is ignored for scoring. Duplicate and unknown",
        "candidates remain in the output and therefore prevent an exact image-set match.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Images scored | {totals['images_scored']} |",
        f"| Ground-truth tiles | {totals['real_tiles']} |",
        f"| Detector candidates | {totals['detector_candidates']} |",
        f"| OpenCV-readable candidates | {totals['readable_candidates']} |",
        (
            "| Correct tiles recovered end to end | "
            f"{totals['correct_tiles_recovered']} / {totals['real_tiles']} "
            f"({_percent(totals['correct_tiles_recovered'], totals['real_tiles'])}) |"
        ),
        (
            "| Images with every tile recovered, allowing extras | "
            f"{totals['images_all_recovered_allowing_extras']} / {totals['images_scored']} "
            f"({_percent(totals['images_all_recovered_allowing_extras'], totals['images_scored'])}) |"
        ),
        (
            "| Images with an exact output set | "
            f"{totals['images_exact_output_set']} / {totals['images_scored']} "
            f"({_percent(totals['images_exact_output_set'], totals['images_scored'])}) |"
        ),
        (
            "| Candidate-level posture labels correct | "
            f"{totals['posture_correct_candidates']} / {totals['detector_candidates']} "
            f"({_percent(totals['posture_correct_candidates'], totals['detector_candidates'])}) |"
        ),
        "",
        "Candidate-level posture accuracy is diagnostic only because duplicate and merged",
        "detections are counted as candidates rather than matched physical instances.",
        "",
        "## Per-image results",
        "",
        "| Image | Expected | Readable predictions | Recovered | Candidates | Exact |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in score["images"]:
        expected = ", ".join(item["expected"])
        predicted = ", ".join(item["predicted"]) or "?"
        exact = "yes" if item["exact_output_set"] else "no"
        lines.append(
            f"| {item['image']} | {expected} | {predicted} | "
            f"{item['recovered']}/{item['real_tiles']} | {item['candidates']} | {exact} |"
        )
    if score["missing_images"]:
        lines.extend(["", "Missing results: " + ", ".join(score["missing_images"])])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    try:
        results = _load_results(args.results)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    score = evaluate(ground_truth, results)
    report = render_report(score)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        args.output.with_suffix(".json").write_text(
            json.dumps(score, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if score["missing_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

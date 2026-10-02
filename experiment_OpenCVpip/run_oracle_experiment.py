#!/usr/bin/env python3
"""Evaluate OpenCV pip counting on manually annotated, detector-free crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiment_OpenCVpip.opencv_pip import (
    OpenCVPipResult,
    RectifiedDominoCrop,
    recognize_rectified_crop,
)


DEFAULT_ANNOTATIONS = Path(
    "experiment_OpenCVpip/manual_annotations/single_domino/annotations.json"
)
DEFAULT_CROPS = Path(
    "experiment_OpenCVpip/manual_annotations/single_domino/crops"
)
DEFAULT_OUTPUT = Path("experiment_OpenCVpip/results/oracle-single")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score OpenCV pip counting on human-drawn oracle crops."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pip-conf", type=float, default=0.22)
    return parser


def _load_crop(path: Path) -> RectifiedDominoCrop:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read crop: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    split_col = image_rgb.shape[1] // 2
    return RectifiedDominoCrop(
        full_crop_rgb=image_rgb,
        end_0_crop_rgb=image_rgb[:, :split_col].copy(),
        end_1_crop_rgb=image_rgb[:, split_col:].copy(),
        split_col=split_col,
    )


def _half_payload(debug: Any, expected: int, min_confidence: float) -> dict[str, Any]:
    accepted = bool(debug.is_valid and debug.confidence >= min_confidence)
    raw_count = int(debug.count)
    return {
        "expected": int(expected),
        "raw_count": raw_count,
        "accepted_count": raw_count if accepted else None,
        "confidence": float(debug.confidence),
        "accepted": accepted,
        "correct_if_accepted": bool(accepted and raw_count == int(expected)),
        "raw_count_correct": bool(raw_count == int(expected)),
        "failure_reason": None if accepted else (debug.failure_reason or "low-confidence"),
    }


def _render_debug(
    crop: RectifiedDominoCrop,
    result: OpenCVPipResult,
    expected: tuple[int, int],
) -> np.ndarray:
    full = cv2.cvtColor(crop.full_crop_rgb, cv2.COLOR_RGB2BGR)
    left = cv2.cvtColor(result.end_0_debug.overlay_rgb, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(result.end_1_debug.overlay_rgb, cv2.COLOR_RGB2BGR)
    target_height = 300
    full = cv2.resize(full, (600, target_height), interpolation=cv2.INTER_AREA)
    left = cv2.resize(left, (300, target_height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (300, target_height), interpolation=cv2.INTER_AREA)
    halves = np.concatenate([left, right], axis=1)
    canvas = np.full((660, 620, 3), 245, dtype=np.uint8)
    canvas[45:345, 10:610] = full
    canvas[350:650, 10:610] = halves
    raw = (result.end_0_debug.count, result.end_1_debug.count)
    cv2.putText(
        canvas,
        f"expected {expected[0]}-{expected[1]} | raw OpenCV {raw[0]}-{raw[1]} | final {result.face_label}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _render_report(rows: list[dict[str, Any]], pip_conf: float) -> str:
    total_tiles = len(rows)
    total_halves = total_tiles * 2
    accepted_halves = sum(
        half["accepted"] for row in rows for half in (row["end_0"], row["end_1"])
    )
    correct_accepted_halves = sum(
        half["correct_if_accepted"]
        for row in rows
        for half in (row["end_0"], row["end_1"])
    )
    raw_correct_halves = sum(
        half["raw_count_correct"]
        for row in rows
        for half in (row["end_0"], row["end_1"])
    )
    readable_tiles = sum(row["readable"] for row in rows)
    correct_tiles = sum(row["tile_correct"] for row in rows)

    def pct(numerator: int, denominator: int) -> str:
        return "n/a" if denominator == 0 else f"{100.0 * numerator / denominator:.1f}%"

    lines = [
        "# Manual-crop OpenCV pip experiment",
        "",
        "This experiment bypasses standing/fallen detection completely. Every input is a",
        "human-drawn, perspective-rectified crop containing exactly one domino.",
        "",
        f"Minimum accepted half confidence: `{pip_conf:.2f}`.",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Manually cropped tiles | {total_tiles} |",
        f"| Total half faces | {total_halves} |",
        f"| Accepted half faces | {accepted_halves} / {total_halves} ({pct(accepted_halves, total_halves)}) |",
        f"| Correct accepted half faces | {correct_accepted_halves} / {total_halves} ({pct(correct_accepted_halves, total_halves)}) |",
        f"| Raw half counts correct before confidence gating | {raw_correct_halves} / {total_halves} ({pct(raw_correct_halves, total_halves)}) |",
        f"| Fully readable tiles | {readable_tiles} / {total_tiles} ({pct(readable_tiles, total_tiles)}) |",
        f"| Fully correct tiles | {correct_tiles} / {total_tiles} ({pct(correct_tiles, total_tiles)}) |",
        "",
        "## Per-image results",
        "",
        "| Image | Expected | Raw OpenCV | Accepted output | Half confidence | Result |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        expected = f"{row['expected'][0]}-{row['expected'][1]}"
        raw = f"{row['end_0']['raw_count']}-{row['end_1']['raw_count']}"
        accepted = row["face_label"]
        confidence = f"{row['end_0']['confidence']:.2f} / {row['end_1']['confidence']:.2f}"
        status = "correct" if row["tile_correct"] else ("rejected" if not row["readable"] else "wrong")
        lines.append(
            f"| {row['image']} | {expected} | {raw} | {accepted} | {confidence} | {status} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation thresholds",
            "",
            "- Raw half accuracy below 85%: prefer a trained 0-6 half-face classifier.",
            "- Raw half accuracy from 85% to 95%: OpenCV may be viable after targeted fixes.",
            "- Raw half accuracy above 95%: continue with OpenCV and then fix detector geometry.",
            "",
            "These ten easy standing tiles are a feasibility screen, not a final robot-camera benchmark.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 <= args.pip_conf <= 1.0:
        raise SystemExit("--pip-conf must be between 0 and 1")
    if not args.annotations.is_file():
        raise SystemExit(f"Annotations not found: {args.annotations}")
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_name, entry in sorted(payload.get("images", {}).items()):
        dominoes = entry.get("dominoes", [])
        if len(dominoes) != 1:
            raise SystemExit(f"Expected exactly one annotation for {image_name}")
        annotation = dominoes[0]
        expected = (int(annotation["pips"][0]), int(annotation["pips"][1]))
        crop_path = args.crops_dir / f"{Path(image_name).stem}-domino-0.png"
        crop = _load_crop(crop_path)
        result = recognize_rectified_crop(
            crop,
            domino_id="domino-0",
            posture=str(annotation["posture"]),
            min_half_confidence=args.pip_conf,
        )
        end_0 = _half_payload(result.end_0_debug, expected[0], args.pip_conf)
        end_1 = _half_payload(result.end_1_debug, expected[1], args.pip_conf)
        row = {
            "image": image_name,
            "crop": str(crop_path),
            "expected": list(expected),
            "face_label": result.face_label,
            "readable": result.readable,
            "tile_correct": bool(
                result.readable
                and result.end_0_pips == expected[0]
                and result.end_1_pips == expected[1]
            ),
            "end_0": end_0,
            "end_1": end_1,
        }
        rows.append(row)
        debug_image = _render_debug(crop, result, expected)
        cv2.imwrite(
            str(debug_dir / f"{Path(image_name).stem}-debug.png"), debug_image
        )

    report = _render_report(rows, args.pip_conf)
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (args.output_dir / "results.json").write_text(
        json.dumps({"pip_confidence_threshold": args.pip_conf, "images": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

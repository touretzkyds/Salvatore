#!/usr/bin/env python3
"""Run standing/fallen detection followed by local OpenCV pip counting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiment_OpenCVpip.opencv_pip import OpenCVPipResult, recognize_observation


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect domino candidates with bestieee.pt/fallen.pt, then count "
            "each half's pips locally with OpenCV."
        )
    )
    parser.add_argument("input", type=Path, help="One image or a directory of images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_OpenCVpip/results"),
    )
    parser.add_argument("--standing-weights", default="bestieee.pt")
    parser.add_argument("--fallen-weights", default="fallen.pt")
    parser.add_argument("--detector-conf", type=float, default=0.35)
    parser.add_argument("--pip-conf", type=float, default=0.22)
    parser.add_argument(
        "--save-crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save rectified crops and per-half OpenCV overlays",
    )
    return parser


def _input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and ".opencv-pip" not in path.stem
        )
    raise ValueError(f"Input does not exist: {input_path}")


def _draw_result(
    image_rgb: np.ndarray,
    observations: list[Any],
    pip_results: list[OpenCVPipResult],
) -> np.ndarray:
    canvas = image_rgb.copy()
    for observation, result in zip(observations, pip_results):
        color = (255, 150, 30) if observation.is_fallen else (30, 190, 255)
        if not result.readable:
            color = (255, 80, 80)
        quad = np.asarray(observation.quad, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [quad], True, color, 4, lineType=cv2.LINE_AA)
        center = tuple(int(round(value)) for value in observation.center_xy)
        text = (
            f"{result.domino_id} {result.face_label} {result.posture} "
            f"det={observation.confidence:.2f} pip={result.confidence:.2f}"
        )
        cv2.putText(
            canvas,
            text,
            (max(0, center[0] - 100), max(24, center[1] - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _serialize_observation(observation: Any, result: OpenCVPipResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload.update(
        {
            "detector_confidence": float(observation.confidence),
            "center_xy": [float(value) for value in observation.center_xy],
            "quad": [[float(x), float(y)] for x, y in observation.quad],
            "axis_endpoints": [
                [float(x), float(y)] for x, y in observation.axis_endpoints
            ],
        }
    )
    return payload


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Could not write image: {path}")


def _save_crops(output_dir: Path, stem: str, results: list[OpenCVPipResult]) -> None:
    crop_dir = output_dir / "crops" / stem
    for result in results:
        base = crop_dir / result.domino_id
        _write_rgb(base.with_name(f"{base.name}-full.png"), result.crop.full_crop_rgb)
        _write_rgb(base.with_name(f"{base.name}-end-0.png"), result.crop.end_0_crop_rgb)
        _write_rgb(base.with_name(f"{base.name}-end-1.png"), result.crop.end_1_crop_rgb)
        _write_rgb(
            base.with_name(f"{base.name}-end-0-debug.png"),
            result.end_0_debug.overlay_rgb,
        )
        _write_rgb(
            base.with_name(f"{base.name}-end-1-debug.png"),
            result.end_1_debug.overlay_rgb,
        )


def _process_image(
    image_path: Path,
    output_dir: Path,
    detector: Any,
    pip_conf: float,
    save_crops: bool,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    observations = detector.detect(image_rgb, frame_id=None)
    results = [
        recognize_observation(
            image_rgb,
            observation,
            domino_id=f"domino-{index}",
            min_half_confidence=pip_conf,
        )
        for index, observation in enumerate(observations)
    ]

    payload = {
        "input_image": str(image_path.resolve()),
        "pipeline": "standing-fallen-detector-plus-opencv-pip-counting",
        "dominoes": [
            _serialize_observation(observation, result)
            for observation, result in zip(observations, results)
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{image_path.stem}.opencv-pip.json"
    image_output_path = output_dir / f"{image_path.stem}.opencv-pip.png"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_rgb(image_output_path, _draw_result(image_rgb, observations, results))
    if save_crops:
        _save_crops(output_dir, image_path.stem, results)
    return payload


def main() -> int:
    args = _parser().parse_args()
    if not 0.0 <= args.detector_conf <= 1.0:
        raise SystemExit("--detector-conf must be between 0 and 1")
    if not 0.0 <= args.pip_conf <= 1.0:
        raise SystemExit("--pip-conf must be between 0 and 1")
    try:
        images = _input_images(args.input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not images:
        raise SystemExit(f"No JPG or PNG images found in: {args.input}")

    # Delay this import because loading torch/ultralytics is expensive.
    from aim_fsm.domino import DominoWorldDetector

    detector = DominoWorldDetector(
        conf_threshold=args.detector_conf,
        standing_weights=args.standing_weights,
        fallen_weights=args.fallen_weights,
        label_mode="none",
    )
    summary = []
    for image_path in images:
        try:
            payload = _process_image(
                image_path=image_path,
                output_dir=args.output_dir,
                detector=detector,
                pip_conf=args.pip_conf,
                save_crops=args.save_crops,
            )
            dominoes = payload["dominoes"]
            readable = sum(bool(item["readable"]) for item in dominoes)
            summary.append(
                {
                    "image": image_path.name,
                    "error": None,
                    "detected": len(dominoes),
                    "readable": readable,
                }
            )
            print(f"{image_path.name}: detected={len(dominoes)} readable={readable}")
        except Exception as exc:
            summary.append(
                {
                    "image": image_path.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "detected": 0,
                    "readable": 0,
                }
            )
            print(f"{image_path.name}: ERROR {type(exc).__name__}: {exc}")

    summary_payload = {
        "input": str(args.input.resolve()),
        "detector": {
            "standing_weights": args.standing_weights,
            "fallen_weights": args.fallen_weights,
            "confidence_threshold": args.detector_conf,
        },
        "opencv": {"minimum_half_confidence": args.pip_conf},
        "images": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    failures = sum(item["error"] is not None for item in summary)
    print(f"Results: {args.output_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

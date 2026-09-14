#!/usr/bin/env python3
"""Run standing/fallen detection plus GPT pip counting on one camera image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from experiments.domino_gpt.recognizer import (
    DEFAULT_DETAIL,
    DEFAULT_MODEL,
    DominoGPTRecognizer,
    DominoPipResult,
    recognize_observations,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect domino bounding boxes/posture with the existing standing and "
            "fallen weights, then classify pips with GPT-5.6 Sol."
        )
    )
    parser.add_argument("image", type=Path, help="Input JPG/PNG camera frame")
    parser.add_argument("--output-image", type=Path, help="Annotated PNG output path")
    parser.add_argument("--output-json", type=Path, help="Machine-readable JSON output path")
    parser.add_argument("--save-crops", type=Path, help="Optional directory for GPT input crops")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI vision model")
    parser.add_argument(
        "--detail",
        default=DEFAULT_DETAIL,
        choices=("low", "high", "original", "auto"),
        help="OpenAI image detail level",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Independent GPT calls; values above 1 require a strict majority",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--standing-weights", default="bestieee.pt")
    parser.add_argument("--fallen-weights", default="fallen.pt")
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Run local detection/cropping without making an OpenAI API request",
    )
    return parser


def _derived_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}.domino-gpt{suffix}")


def _draw_observations(image_rgb: np.ndarray, observations: list[Any]) -> np.ndarray:
    canvas = image_rgb.copy()
    for index, observation in enumerate(observations):
        color = (255, 150, 30) if bool(getattr(observation, "is_fallen", False)) else (30, 190, 255)
        quad = np.asarray(observation.quad, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [quad], True, color, 3, lineType=cv2.LINE_AA)

        endpoint_0 = tuple(int(round(value)) for value in observation.axis_endpoints[0])
        endpoint_1 = tuple(int(round(value)) for value in observation.axis_endpoints[1])
        cv2.arrowedLine(canvas, endpoint_0, endpoint_1, (255, 255, 255), 2, tipLength=0.12)
        cv2.circle(canvas, endpoint_0, 7, (255, 80, 80), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, endpoint_1, 7, (80, 255, 80), -1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, "0", endpoint_0, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        cv2.putText(canvas, "1", endpoint_1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)

        center = tuple(int(round(value)) for value in observation.center_xy)
        posture = "fallen" if bool(getattr(observation, "is_fallen", False)) else "standing"
        label = getattr(observation, "face_label", None) or "?"
        text = f"D{index} {label} {posture}"
        origin = (max(0, center[0] - 55), max(18, center[1] - 24))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def _serialize_observation(
    index: int,
    observation: Any,
    prediction: Optional[DominoPipResult],
) -> dict[str, Any]:
    counts = getattr(observation, "half_counts", (None, None)) or (None, None)
    return {
        "domino_id": f"domino-{index}",
        "posture": "fallen" if bool(getattr(observation, "is_fallen", False)) else "standing",
        "detector_confidence": float(observation.confidence),
        "center_xy": [float(value) for value in observation.center_xy],
        "quad": [[float(x), float(y)] for x, y in observation.quad],
        "axis_endpoints": [
            [float(x), float(y)] for x, y in observation.axis_endpoints
        ],
        "end_0_pips": counts[0],
        "end_1_pips": counts[1],
        "readable": bool(prediction.readable) if prediction is not None else False,
        "gpt_agreement": float(prediction.agreement) if prediction is not None else None,
        "uncertainty_reason": prediction.uncertainty_reason if prediction is not None else None,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if not args.image.is_file():
        raise SystemExit(f"Input image does not exist: {args.image}")
    if not args.detection_only and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required unless --detection-only is used.")

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"OpenCV could not read: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Importing this module loads torch/ultralytics, so keep it after argument
    # and input validation. The detector is explicitly placed in geometry-only
    # mode; the original CNN implementation remains its default mode.
    from aim_fsm.domino import DominoWorldDetector

    detector = DominoWorldDetector(
        conf_threshold=args.conf,
        standing_weights=args.standing_weights,
        fallen_weights=args.fallen_weights,
        label_mode="none",
    )
    observations = detector.detect(image_rgb, frame_id=0)
    predictions: dict[str, DominoPipResult] = {}
    crops = []

    if observations and not args.detection_only:
        recognizer = DominoGPTRecognizer(
            model=args.model,
            detail=args.detail,
            reasoning_effort=args.reasoning_effort,
        )
        observations, predictions, crops = recognize_observations(
            image_rgb,
            observations,
            recognizer,
            samples=args.samples,
        )
    elif observations:
        from experiments.domino_gpt.recognizer import build_domino_crop

        crops = [
            build_domino_crop(image_rgb, observation, f"domino-{index}")
            for index, observation in enumerate(observations)
        ]

    output_image = args.output_image or _derived_output_path(args.image, ".png")
    output_json = args.output_json or _derived_output_path(args.image, ".json")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    annotated_rgb = _draw_observations(image_rgb, observations)
    if not cv2.imwrite(str(output_image), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)):
        raise SystemExit(f"Could not write annotated image: {output_image}")

    payload = {
        "input_image": str(args.image.resolve()),
        "detector": {
            "standing_weights": args.standing_weights,
            "fallen_weights": args.fallen_weights,
            "confidence_threshold": args.conf,
        },
        "gpt": None
        if args.detection_only
        else {
            "model": args.model,
            "detail": args.detail,
            "reasoning_effort": args.reasoning_effort,
            "samples": args.samples,
        },
        "dominoes": [
            _serialize_observation(index, observation, predictions.get(f"domino-{index}"))
            for index, observation in enumerate(observations)
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)
        for crop in crops:
            for name, image in (
                ("full", crop.full_crop_rgb),
                ("end-0", crop.end_0_crop_rgb),
                ("end-1", crop.end_1_crop_rgb),
            ):
                crop_path = args.save_crops / f"{crop.domino_id}-{name}.png"
                cv2.imwrite(str(crop_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(json.dumps(payload, indent=2))
    print(f"Annotated image: {output_image}")
    print(f"JSON result: {output_json}")
    if not observations:
        print("No dominoes were detected in this frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run detector-free GPT domino recognition on one complete camera image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2

from experiments.domino_gpt.direct_recognizer import DirectDominoGPTRecognizer
from experiments.domino_gpt.recognizer import DEFAULT_DETAIL, DEFAULT_MODEL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and read dominoes directly with GPT; no .pt models are loaded."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-image", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--detail", default=DEFAULT_DETAIL, choices=("low", "high", "original", "auto")
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
    )
    return parser


def _draw(image_bgr, dominoes):
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    colors = {
        "standing": (255, 180, 30),
        "fallen": (30, 140, 255),
        "uncertain": (80, 80, 255),
    }
    for item in dominoes:
        left, top, right, bottom = item.bbox_normalized
        x0 = round(left * width / 1000)
        y0 = round(top * height / 1000)
        x1 = round(right * width / 1000)
        y1 = round(bottom * height / 1000)
        color = colors[item.posture]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 4)
        label = f"{item.domino_id} {item.face_label} {item.posture}"
        cv2.putText(
            canvas,
            label,
            (max(0, x0), max(28, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    args = _parser().parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Input image does not exist: {args.image}")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"OpenCV could not read: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    recognizer = DirectDominoGPTRecognizer(
        model=args.model,
        detail=args.detail,
        reasoning_effort=args.reasoning_effort,
    )
    dominoes = recognizer.recognize(image_rgb)

    output_image = args.output_image or args.image.with_name(
        f"{args.image.stem}.direct-gpt.png"
    )
    output_json = args.output_json or args.image.with_name(
        f"{args.image.stem}.direct-gpt.json"
    )
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "input_image": str(args.image.resolve()),
        "pipeline": "direct-full-frame-gpt-no-local-detector",
        "gpt": {
            "model": args.model,
            "detail": args.detail,
            "reasoning_effort": args.reasoning_effort,
        },
        "dominoes": [item.to_dict() for item in dominoes],
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    cv2.imwrite(str(output_image), _draw(image_bgr, dominoes))
    print(json.dumps(payload, indent=2))
    print(f"Annotated image: {output_image}")
    print(f"JSON result: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

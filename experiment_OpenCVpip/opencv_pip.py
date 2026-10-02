"""OpenCV pip counting for domino geometry produced by standing/fallen models.

This module deliberately does not run a detector.  It receives one detector
observation, perspective-rectifies its quadrilateral, splits the tile along its
geometric midpoint, and applies the repository's classical OpenCV pip counter
to each half.

Keeping this adapter separate makes it possible to score detector geometry and
pip recognition independently.  A merged or duplicate detector box remains
visible in the experiment output instead of being silently hidden here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from lab8.domino_world_detector import PipCountDebug, analyze_half_crop


DEFAULT_CROP_SIZE = (512, 256)


@dataclass(frozen=True)
class RectifiedDominoCrop:
    full_crop_rgb: np.ndarray
    end_0_crop_rgb: np.ndarray
    end_1_crop_rgb: np.ndarray
    split_col: int


@dataclass(frozen=True)
class OpenCVPipResult:
    domino_id: str
    posture: str
    end_0_pips: Optional[int]
    end_1_pips: Optional[int]
    end_0_confidence: float
    end_1_confidence: float
    readable: bool
    uncertainty_reason: Optional[str]
    crop: RectifiedDominoCrop
    end_0_debug: PipCountDebug
    end_1_debug: PipCountDebug

    @property
    def face_label(self) -> str:
        if not self.readable:
            return "?"
        return f"{self.end_0_pips}-{self.end_1_pips}"

    @property
    def confidence(self) -> float:
        if not self.readable:
            return 0.0
        return float(min(self.end_0_confidence, self.end_1_confidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "domino_id": self.domino_id,
            "posture": self.posture,
            "face_label": self.face_label,
            "end_0_pips": self.end_0_pips,
            "end_1_pips": self.end_1_pips,
            "end_0_confidence": self.end_0_confidence,
            "end_1_confidence": self.end_1_confidence,
            "confidence": self.confidence,
            "readable": self.readable,
            "uncertainty_reason": self.uncertainty_reason,
            "split_col": self.crop.split_col,
        }


def _order_quad_along_directed_axis(
    quad: np.ndarray,
    center_xy: Sequence[float],
    axis_endpoints: Sequence[Sequence[float]],
) -> np.ndarray:
    """Order corners so output left is detector endpoint 0 and right is 1."""
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = np.asarray(center_xy, dtype=np.float32).reshape(2)
    endpoint_0 = np.asarray(axis_endpoints[0], dtype=np.float32).reshape(2)
    endpoint_1 = np.asarray(axis_endpoints[1], dtype=np.float32).reshape(2)
    long_dir = endpoint_1 - endpoint_0
    norm = float(np.linalg.norm(long_dir))
    if norm <= 1e-6:
        raise ValueError("The domino axis endpoints collapse to one point.")
    long_dir /= norm
    short_dir = np.array([-long_dir[1], long_dir[0]], dtype=np.float32)

    enriched = []
    for point in points:
        delta = point - center
        enriched.append(
            (point, float(np.dot(delta, long_dir)), float(np.dot(delta, short_dir)))
        )
    enriched.sort(key=lambda item: item[1])
    end_0_corners = sorted(enriched[:2], key=lambda item: item[2])
    end_1_corners = sorted(enriched[2:], key=lambda item: item[2])
    return np.array(
        [
            end_0_corners[0][0],
            end_1_corners[0][0],
            end_1_corners[1][0],
            end_0_corners[1][0],
        ],
        dtype=np.float32,
    )


def rectify_and_split(
    image_rgb: np.ndarray,
    observation: Any,
    output_size: tuple[int, int] = DEFAULT_CROP_SIZE,
) -> RectifiedDominoCrop:
    """Perspective-rectify a detector quad and split it at the midpoint."""
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot crop an empty camera image.")
    width, height = (int(output_size[0]), int(output_size[1]))
    if width < 2 or height < 2:
        raise ValueError("output_size must be at least 2 by 2 pixels.")

    src = _order_quad_along_directed_axis(
        quad=np.asarray(observation.quad, dtype=np.float32),
        center_xy=observation.center_xy,
        axis_endpoints=observation.axis_endpoints,
    )
    dst = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    full_crop = cv2.warpPerspective(
        image_rgb,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    split_col = width // 2
    return RectifiedDominoCrop(
        full_crop_rgb=full_crop,
        end_0_crop_rgb=full_crop[:, :split_col].copy(),
        end_1_crop_rgb=full_crop[:, split_col:].copy(),
        split_col=split_col,
    )


def recognize_rectified_crop(
    crop: RectifiedDominoCrop,
    domino_id: str,
    posture: str,
    min_half_confidence: float = 0.22,
) -> OpenCVPipResult:
    """Count both halves of an already rectified domino crop."""
    end_0_debug = analyze_half_crop(crop.end_0_crop_rgb, side="end-0")
    end_1_debug = analyze_half_crop(crop.end_1_crop_rgb, side="end-1")
    end_0_ok = end_0_debug.is_valid and end_0_debug.confidence >= min_half_confidence
    end_1_ok = end_1_debug.is_valid and end_1_debug.confidence >= min_half_confidence
    readable = bool(end_0_ok and end_1_ok)

    reasons = []
    if not end_0_ok:
        reasons.append(
            f"end-0:{end_0_debug.failure_reason or 'low-confidence'}"
        )
    if not end_1_ok:
        reasons.append(
            f"end-1:{end_1_debug.failure_reason or 'low-confidence'}"
        )

    return OpenCVPipResult(
        domino_id=str(domino_id),
        posture=str(posture),
        end_0_pips=int(end_0_debug.count) if readable else None,
        end_1_pips=int(end_1_debug.count) if readable else None,
        end_0_confidence=float(end_0_debug.confidence),
        end_1_confidence=float(end_1_debug.confidence),
        readable=readable,
        uncertainty_reason="; ".join(reasons) if reasons else None,
        crop=crop,
        end_0_debug=end_0_debug,
        end_1_debug=end_1_debug,
    )


def recognize_observation(
    image_rgb: np.ndarray,
    observation: Any,
    domino_id: str,
    min_half_confidence: float = 0.22,
) -> OpenCVPipResult:
    crop = rectify_and_split(image_rgb, observation)
    posture = "fallen" if bool(getattr(observation, "is_fallen", False)) else "standing"
    return recognize_rectified_crop(
        crop,
        domino_id=domino_id,
        posture=posture,
        min_half_confidence=min_half_confidence,
    )

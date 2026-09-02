from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


WHITE = (255, 255, 255)
DEFAULT_FULL_WIDTH = 120
DEFAULT_FULL_HEIGHT = 60
DEFAULT_HALF_SIZE = 64
NONWHITE_CROP_MARGIN = 4


@dataclass(frozen=True)
class RectifiedDominoResult:
    full_crop_rgb: np.ndarray
    rectified_mask: np.ndarray
    transform: np.ndarray
    inverse_transform: np.ndarray
    rectified_divider: Optional[np.ndarray]
    divider_found: bool


@dataclass(frozen=True)
class HalfFaceCropResult:
    left_half_rgb: np.ndarray
    right_half_rgb: np.ndarray
    split_col: int
    divider_x: float
    half_image_centers: tuple[tuple[float, float], tuple[float, float]]


def order_quad_from_axis(observation: Any) -> np.ndarray:
    theta = float(observation.axis_theta)
    long_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    short_dir = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    center = np.array(observation.center_xy, dtype=np.float32)

    enriched = []
    for pt in observation.quad:
        point = np.array(pt, dtype=np.float32)
        vec = point - center
        long_coord = float(np.dot(vec, long_dir))
        short_coord = float(np.dot(vec, short_dir))
        enriched.append((point, long_coord, short_coord))

    enriched.sort(key=lambda item: item[1])
    left = sorted(enriched[:2], key=lambda item: item[2])
    right = sorted(enriched[2:], key=lambda item: item[2])
    if len(left) != 2 or len(right) != 2:
        raise ValueError("Could not order quad corners from the detector observation.")

    return np.array([left[0][0], right[0][0], right[1][0], left[1][0]], dtype=np.float32)


def letterbox_square(image: np.ndarray, size: int = DEFAULT_HALF_SIZE) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot letterbox an empty image.")

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x0 = (size - resized_w) // 2
    y0 = (size - resized_h) // 2
    canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return canvas


def crop_to_nonwhite_bounds(image: np.ndarray, margin: int = NONWHITE_CROP_MARGIN) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot crop an empty image.")

    keep_mask = np.any(image < 250, axis=2)
    if not np.any(keep_mask):
        return image.copy()

    ys, xs = np.where(keep_mask)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y0:y1, x0:x1].copy()


def rectify_domino(
    image_rgb: np.ndarray,
    observation: Any,
    full_width: int = DEFAULT_FULL_WIDTH,
    full_height: int = DEFAULT_FULL_HEIGHT,
) -> RectifiedDominoResult:
    src = order_quad_from_axis(observation)
    dst = np.array(
        [
            [0.0, 0.0],
            [full_width - 1.0, 0.0],
            [full_width - 1.0, full_height - 1.0],
            [0.0, full_height - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    inverse_transform = cv2.getPerspectiveTransform(dst, src)

    rectified_rgb = cv2.warpPerspective(
        image_rgb,
        transform,
        (full_width, full_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=WHITE,
    )
    rectified_mask = cv2.warpPerspective(
        observation.mask,
        transform,
        (full_width, full_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rectified_mask = (rectified_mask > 0).astype(np.uint8) * 255
    rectified_rgb = np.where(rectified_mask[..., None] > 0, rectified_rgb, 255).astype(np.uint8)

    divider_found = observation.divider_endpoints is not None
    rectified_divider = None
    if divider_found:
        divider_pts = np.array(observation.divider_endpoints, dtype=np.float32).reshape(-1, 1, 2)
        rectified_divider = cv2.perspectiveTransform(divider_pts, transform).reshape(-1, 2)

    return RectifiedDominoResult(
        full_crop_rgb=rectified_rgb,
        rectified_mask=rectified_mask,
        transform=transform,
        inverse_transform=inverse_transform,
        rectified_divider=rectified_divider,
        divider_found=divider_found,
    )


def _mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Cannot compute a center for an empty mask.")
    return (float(xs.mean()), float(ys.mean()))


def _map_rectified_point_to_image(point_xy: tuple[float, float], inverse_transform: np.ndarray) -> tuple[float, float]:
    points = np.array([[point_xy]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points, inverse_transform)[0][0]
    return (float(mapped[0]), float(mapped[1]))


def split_rectified_halves(
    rectified: RectifiedDominoResult,
    half_size: int = DEFAULT_HALF_SIZE,
) -> HalfFaceCropResult:
    if rectified.rectified_divider is None:
        raise ValueError("Cannot split halves without a detected divider line.")

    rectified_rgb = rectified.full_crop_rgb
    rectified_mask = rectified.rectified_mask
    rectified_divider = rectified.rectified_divider

    height, width = rectified_rgb.shape[:2]
    p0 = rectified_divider[0].astype(np.float32)
    p1 = rectified_divider[1].astype(np.float32)
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        raise ValueError("Divider line collapsed in rectified coordinates.")

    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    signed = (xx - p0[0]) * direction[1] - (yy - p0[1]) * direction[0]
    domino_mask = rectified_mask > 0

    left_probe = np.array([0.25 * (width - 1), 0.5 * (height - 1)], dtype=np.float32)
    right_probe = np.array([0.75 * (width - 1), 0.5 * (height - 1)], dtype=np.float32)
    left_probe_sign = float((left_probe[0] - p0[0]) * direction[1] - (left_probe[1] - p0[1]) * direction[0])
    right_probe_sign = float((right_probe[0] - p0[0]) * direction[1] - (right_probe[1] - p0[1]) * direction[0])

    if left_probe_sign == 0.0:
        left_probe_sign = -1.0
    if right_probe_sign == 0.0:
        right_probe_sign = 1.0

    left_mask = domino_mask & (signed * left_probe_sign >= 0.0)
    right_mask = domino_mask & (signed * right_probe_sign >= 0.0)
    if not left_mask.any() or not right_mask.any():
        raise ValueError("Divider split removed one half entirely.")

    left_full = np.where(left_mask[..., None], rectified_rgb, 255).astype(np.uint8)
    right_full = np.where(right_mask[..., None], rectified_rgb, 255).astype(np.uint8)
    left = letterbox_square(crop_to_nonwhite_bounds(left_full), half_size)
    right = letterbox_square(crop_to_nonwhite_bounds(right_full), half_size)

    left_center_rectified = _mask_center(left_mask)
    right_center_rectified = _mask_center(right_mask)
    half_image_centers = (
        _map_rectified_point_to_image(left_center_rectified, rectified.inverse_transform),
        _map_rectified_point_to_image(right_center_rectified, rectified.inverse_transform),
    )

    split_col = int(round(float(rectified_divider[:, 0].mean())))
    split_col = max(1, min(width - 1, split_col))
    divider_x = float(rectified_divider[:, 0].mean())

    return HalfFaceCropResult(
        left_half_rgb=left,
        right_half_rgb=right,
        split_col=split_col,
        divider_x=divider_x,
        half_image_centers=half_image_centers,
    )


def extract_half_face_crops(
    image_rgb: np.ndarray,
    observation: Any,
    full_width: int = DEFAULT_FULL_WIDTH,
    full_height: int = DEFAULT_FULL_HEIGHT,
    half_size: int = DEFAULT_HALF_SIZE,
) -> tuple[RectifiedDominoResult, Optional[HalfFaceCropResult]]:
    rectified = rectify_domino(
        image_rgb=image_rgb,
        observation=observation,
        full_width=full_width,
        full_height=full_height,
    )
    if not rectified.divider_found:
        return rectified, None
    halves = split_rectified_halves(rectified, half_size=half_size)
    return rectified, halves

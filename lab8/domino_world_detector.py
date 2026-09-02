from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Optional

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from lab8.domino_half_face_pipeline import (
        DEFAULT_FULL_HEIGHT as PIPELINE_FULL_HEIGHT,
        DEFAULT_FULL_WIDTH as PIPELINE_FULL_WIDTH,
        DEFAULT_HALF_SIZE,
        crop_to_nonwhite_bounds,
        extract_half_face_crops,
    )
except ImportError:
    from domino_half_face_pipeline import (  # type: ignore
        DEFAULT_FULL_HEIGHT as PIPELINE_FULL_HEIGHT,
        DEFAULT_FULL_WIDTH as PIPELINE_FULL_WIDTH,
        DEFAULT_HALF_SIZE,
        crop_to_nonwhite_bounds,
        extract_half_face_crops,
    )

WEIGHTS_VERSION = "train5"
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weights")
DEFAULT_SEGMENT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "domino_segment.pt")
# Older checkouts kept the segmentation weights inside the ultralytics run directory.
LEGACY_SEGMENT_WEIGHTS_PATHS = (
    os.path.normpath(os.path.join(REPO_ROOT, "runs", "segment", WEIGHTS_VERSION, "weights", "best.pt")),
)
DOMINO_LENGTH_MM = 48.0
DOMINO_WIDTH_MM = 24.0
DOMINO_THICKNESS_MM = 8.0


def resolve_segment_weights(weights_path: Optional[str] = None) -> str:
    """Locate the YOLO segmentation weights, searching weights/ then legacy run dirs.

    Weights are deliberately kept out of git; see weights/README.md.
    """
    if weights_path:
        candidates = [weights_path if os.path.isabs(weights_path)
                      else os.path.normpath(os.path.join(WEIGHTS_DIR, weights_path))]
    else:
        candidates = [DEFAULT_SEGMENT_WEIGHTS_PATH, *LEGACY_SEGMENT_WEIGHTS_PATHS]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    searched = "\n  ".join(candidates)
    raise FileNotFoundError(
        "Missing domino segmentation weights. Looked in:\n  "
        f"{searched}\nSee weights/README.md for how to obtain them."
    )


def roboflow_fit_resize(img: np.ndarray, size: int = 160) -> tuple[np.ndarray, float, int, int, int, int]:
    """Match the training-time letterbox transform before running YOLO."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y, new_w, new_h


def map_mask_to_original(
    mask_reduced: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Undo the letterbox transform so masks line up with the original camera image."""
    cropped = mask_reduced[
        pad_y:pad_y + int(round(orig_h * scale)),
        pad_x:pad_x + int(round(orig_w * scale)),
    ]
    restored = cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return (restored > 0).astype(np.uint8) * 255


def normalize_axis_angle(theta: float | None) -> float | None:
    """Treat domino orientation as an axis, so theta and theta + pi are equivalent."""
    if theta is None:
        return None
    return float(theta % math.pi)


def axis_angle_distance(theta_a: float | None, theta_b: float | None) -> float:
    """Angular distance on an axis, not a directed heading."""
    if theta_a is None or theta_b is None:
        return 0.0
    return abs(math.atan2(math.sin(2.0 * (theta_a - theta_b)), math.cos(2.0 * (theta_a - theta_b)))) / 2.0


@dataclass(frozen=True)
class DominoObservation:
    mask: np.ndarray
    contour: np.ndarray
    center_xy: tuple[float, float]
    rect_size: tuple[float, float]
    quad: tuple[tuple[float, float], ...]
    axis_endpoints: tuple[tuple[float, float], tuple[float, float]]
    divider_endpoints: Optional[tuple[tuple[float, float], tuple[float, float]]]
    axis_theta: float
    mask_area: float
    confidence: float
    face_label: Optional[str] = None
    face_confidence: Optional[float] = None
    half_counts: Optional[tuple[Optional[int], Optional[int]]] = None
    half_image_centers: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
    debug_panel: Optional[np.ndarray] = None


class DominoLabelProvider:
    def label(self, image_rgb: np.ndarray, observation: DominoObservation) -> tuple[Optional[str], Optional[float]]:
        return (None, None)

    def annotate(self, image_rgb: np.ndarray, observation: DominoObservation) -> DominoObservation:
        face_label, face_confidence = self.label(image_rgb, observation)
        return replace(observation, face_label=face_label, face_confidence=face_confidence)


class NullDominoLabelProvider(DominoLabelProvider):
    pass


WHITE = (255, 255, 255)

COUNT_COLOR_FAMILY = {
    1: "purple",
    2: "green",
    3: "purple",
    4: "cyan",
    5: "green",
    6: "amber",
}

COLOR_FAMILY_HUE = {
    "green": 58.0,
    "purple": 145.0,
    "cyan": 96.0,
    "amber": 18.0,
}


@dataclass(frozen=True)
class PipCountDebug:
    count: int
    confidence: float
    is_valid: bool
    failure_reason: Optional[str]
    raw_blob_count: int
    layout_score: float
    color_score: float
    color_family: Optional[str]
    response_map: np.ndarray
    anchor_mask: np.ndarray
    template_mask: np.ndarray
    candidate_mask: np.ndarray
    accepted_mask: np.ndarray
    rejected_mask: np.ndarray
    overlay_rgb: np.ndarray


@dataclass(frozen=True)
class DominoBeliefSample:
    center_xy: tuple[float, float]
    counts: tuple[int, int]
    half_confidences: tuple[float, float]
    face_confidence: float
    tick: int


@dataclass(frozen=True)
class DominoBeliefResult:
    counts: tuple[Optional[int], Optional[int]]
    confidences: tuple[float, float]
    is_stable: bool


@dataclass(frozen=True)
class PipBlob:
    contour: np.ndarray
    component_mask: np.ndarray
    center_xy: tuple[float, float]
    normalized_xy: tuple[float, float]
    area_ratio: float
    quality: float
    mean_sat: float
    mean_chroma: float
    mean_hue: float


@dataclass(frozen=True)
class RejectedPipCandidate:
    blob: PipBlob
    reasons: tuple[str, ...]


def _active_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask_u8.shape[1], mask_u8.shape[0])
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _normalize_point(point_xy: tuple[float, float], bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0 - 1)
    height = max(1, y1 - y0 - 1)
    return (
        float(np.clip((point_xy[0] - x0) / width, 0.0, 1.0)),
        float(np.clip((point_xy[1] - y0) / height, 0.0, 1.0)),
    )


def _circular_hue_mean(hues: np.ndarray) -> Optional[float]:
    if hues.size == 0:
        return None
    angles = hues.astype(np.float32) * (2.0 * math.pi / 180.0)
    mean_angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    if mean_angle < 0.0:
        mean_angle += 2.0 * math.pi
    return float(mean_angle * 180.0 / (2.0 * math.pi))


def _hue_distance(hue_a: float, hue_b: float) -> float:
    delta = abs(hue_a - hue_b)
    return min(delta, 180.0 - delta)


def _color_score_for_count(blob_hue: Optional[float], count: int) -> tuple[float, Optional[str]]:
    family = COUNT_COLOR_FAMILY.get(count)
    if family is None or blob_hue is None:
        return (0.5, family)
    expected_hue = COLOR_FAMILY_HUE[family]
    distance = _hue_distance(blob_hue, expected_hue)
    return (float(np.clip(1.0 - distance / 40.0, 0.0, 1.0)), family)


def _blob_center_array(blobs: list[PipBlob]) -> np.ndarray:
    if not blobs:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([blob.normalized_xy for blob in blobs], dtype=np.float32)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _score_two(points: np.ndarray) -> float:
    if len(points) != 2:
        return 0.0
    dx = abs(float(points[1, 0] - points[0, 0]))
    dy = abs(float(points[1, 1] - points[0, 1]))
    sep = math.hypot(dx, dy)
    if sep < 0.25:
        return 0.0
    diagness = min(dx, dy) / max(max(dx, dy), 1e-6)
    return _clip01(0.55 * _clip01((sep - 0.25) / 0.45) + 0.45 * diagness)


def _score_three(points: np.ndarray) -> float:
    if len(points) != 3:
        return 0.0
    best = 0.0
    for idx in range(3):
        center = points[idx]
        pair = np.delete(points, idx, axis=0)
        diag_score = _score_two(pair)
        midpoint = 0.5 * (pair[0] + pair[1])
        center_score = _clip01(1.0 - float(np.linalg.norm(center - midpoint)) / 0.22)
        best = max(best, 0.60 * diag_score + 0.40 * center_score)
    return best


def _quadrant_score(points: np.ndarray, center: np.ndarray) -> float:
    quadrants = {
        (int(point[0] >= center[0]), int(point[1] >= center[1]))
        for point in points
        if abs(float(point[0] - center[0])) > 0.03 and abs(float(point[1] - center[1])) > 0.03
    }
    return len(quadrants) / 4.0


def _score_four(points: np.ndarray) -> float:
    if len(points) != 4:
        return 0.0
    center = points.mean(axis=0)
    spread_x = float(np.ptp(points[:, 0]))
    spread_y = float(np.ptp(points[:, 1]))
    spread_score = _clip01(min(spread_x / 0.35, 1.0) * min(spread_y / 0.35, 1.0))
    return _clip01(0.80 + 0.12 * spread_score + 0.08 * _quadrant_score(points, center))


def _score_five(points: np.ndarray) -> float:
    if len(points) != 5:
        return 0.0
    best = 0.0
    for idx in range(5):
        center = points[idx]
        corners = np.delete(points, idx, axis=0)
        center_score = _clip01(1.0 - float(np.linalg.norm(center - corners.mean(axis=0))) / 0.20)
        quadrant_score = _quadrant_score(corners, center)
        diag_pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                midpoint = 0.5 * (corners[i] + corners[j])
                if np.linalg.norm(midpoint - center) < 0.18:
                    diag_pairs.append((i, j))
        diag_score = _clip01(len(diag_pairs) / 2.0)
        best = max(best, 0.40 * center_score + 0.35 * quadrant_score + 0.25 * diag_score)
    return best


def _score_six(points: np.ndarray) -> float:
    if len(points) != 6:
        return 0.0
    order = np.argsort(points[:, 0])
    left = points[order[:3]]
    right = points[order[3:]]
    if float(left[:, 0].mean()) > float(right[:, 0].mean()):
        left, right = right, left
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    x_sep = float(right[:, 0].mean() - left[:, 0].mean())
    left_spread = float(np.ptp(left[:, 0]))
    right_spread = float(np.ptp(right[:, 0]))
    column_score = _clip01((x_sep - 0.12) / 0.22) * _clip01(1.0 - max(left_spread, right_spread) / 0.18)

    row_diffs = np.abs(left[:, 1] - right[:, 1])
    row_score = _clip01(1.0 - float(np.mean(row_diffs)) / 0.18)

    left_steps = np.diff(left[:, 1])
    right_steps = np.diff(right[:, 1])
    if len(left_steps) != 2 or len(right_steps) != 2:
        return 0.0
    if np.any(left_steps <= 0.04) or np.any(right_steps <= 0.04):
        return 0.0
    top_gap = 0.5 * (left_steps[0] + right_steps[0])
    bottom_gap = 0.5 * (left_steps[1] + right_steps[1])
    perspective_ratio = top_gap / max(bottom_gap, 1e-6)
    perspective_score = _clip01(1.0 - abs(perspective_ratio - 0.85) / 0.75)

    return _clip01(0.42 * column_score + 0.34 * row_score + 0.24 * perspective_score)


def _relationship_score_for_count(blobs: list[PipBlob], count: int) -> float:
    points = _blob_center_array(blobs)
    if count == 0:
        return 1.0 if len(points) == 0 else 0.0
    if count == 1:
        if len(points) != 1:
            return 0.0
        return _clip01(1.0 - float(np.linalg.norm(points[0] - np.array([0.5, 0.5], dtype=np.float32))) / 0.30)
    if count == 2:
        return _score_two(points)
    if count == 3:
        return _score_three(points)
    if count == 4:
        return _score_four(points)
    if count == 5:
        return _score_five(points)
    if count == 6:
        return _score_six(points)
    return 0.0


def _draw_relationship_mask(shape: tuple[int, int], active_mask: np.ndarray, blobs: list[PipBlob], count: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not blobs:
        return mask
    centers = [tuple(int(round(v)) for v in blob.center_xy) for blob in blobs]
    for center in centers:
        cv2.circle(mask, center, 6, 255, 1, lineType=cv2.LINE_AA)

    if count == 5 and len(blobs) == 5:
        pts = _blob_center_array(blobs)
        best_idx = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0, keepdims=True), axis=1)))
        center = centers[best_idx]
        corners = [c for i, c in enumerate(centers) if i != best_idx]
        for corner in corners:
            cv2.line(mask, center, corner, 255, 1, lineType=cv2.LINE_AA)
    elif count == 6 and len(blobs) == 6:
        pts = _blob_center_array(blobs)
        order = np.argsort(pts[:, 0])
        left_idx = order[:3]
        right_idx = order[3:]
        left_pts = [centers[i] for i in left_idx[np.argsort(pts[left_idx, 1])]]
        right_pts = [centers[i] for i in right_idx[np.argsort(pts[right_idx, 1])]]
        for group in (left_pts, right_pts):
            for a, b in zip(group[:-1], group[1:]):
                cv2.line(mask, a, b, 255, 1, lineType=cv2.LINE_AA)
        for a, b in zip(left_pts, right_pts):
            cv2.line(mask, a, b, 255, 1, lineType=cv2.LINE_AA)
    return cv2.bitwise_and(mask, active_mask)


def _candidate_blob_sets(base_blobs: list[PipBlob], repair_blobs: list[PipBlob]) -> list[tuple[int, list[PipBlob]]]:
    base_count = len(base_blobs)
    candidates: list[tuple[int, list[PipBlob]]] = [(min(base_count, 6), list(base_blobs))]
    if base_blobs:
        drop_pool = sorted(range(len(base_blobs)), key=lambda idx: base_blobs[idx].quality)[:4]
        for drop_idx in drop_pool:
            reduced = [blob for idx, blob in enumerate(base_blobs) if idx != drop_idx]
            candidates.append((min(len(reduced), 6), reduced))
    if repair_blobs:
        repair_pool = sorted(repair_blobs, key=lambda blob: blob.quality, reverse=True)[:4]
        max_add = min(2, 6 - min(base_count, 6), len(repair_pool))
        for add_count in range(1, max_add + 1):
            for added in combinations(repair_pool, add_count):
                extended = list(base_blobs) + list(added)
                candidates.append((min(len(extended), 6), extended))

    deduped: list[tuple[int, list[PipBlob]]] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for count, blobs in candidates:
        key = (count, tuple(sorted(id(blob) for blob in blobs)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((count, blobs))
    return deduped


def _color_matched_repair_blobs(accepted_blobs: list[PipBlob], rejected_candidates: list[RejectedPipCandidate]) -> list[PipBlob]:
    color_reference_blobs = [
        blob
        for blob in accepted_blobs
        if blob.quality >= 0.42 and blob.mean_sat >= 42.0 and blob.mean_chroma >= 28.0
    ]
    if len(color_reference_blobs) < 2:
        return []

    reference_hue = _circular_hue_mean(np.array([blob.mean_hue for blob in color_reference_blobs], dtype=np.float32))
    if reference_hue is None:
        return []

    promoted: list[PipBlob] = []
    hard_reject_reasons = {"degenerate", "empty", "large", "elongated", "sparse"}
    for candidate in rejected_candidates:
        if hard_reject_reasons & set(candidate.reasons):
            continue
        hue_distance = _hue_distance(candidate.blob.mean_hue, reference_hue)
        if hue_distance > 18.0:
            continue
        if candidate.blob.mean_chroma < 20.0 or candidate.blob.mean_sat < 22.0:
            continue
        if not (0.0015 <= candidate.blob.area_ratio <= 0.085):
            continue

        color_score = _clip01(1.0 - hue_distance / 18.0)
        promoted.append(
            replace(
                candidate.blob,
                quality=float(np.clip(0.42 + 0.18 * color_score + 0.28 * candidate.blob.quality, 0.0, 0.70)),
            )
        )
    return promoted



def order_quad_from_axis(observation: DominoObservation) -> np.ndarray:
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
        raise ValueError("Could not order quad corners from detector observation.")
    return np.array([left[0][0], right[0][0], right[1][0], left[1][0]], dtype=np.float32)


def rectify_domino(
    image_rgb: np.ndarray,
    observation: DominoObservation,
    full_width: int,
    full_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
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

    divider_found = observation.divider_endpoints is not None
    if divider_found:
        divider_pts = np.array([observation.divider_endpoints], dtype=np.float32)
        rectified_divider = cv2.perspectiveTransform(divider_pts, transform)[0]
        divider_x = float(rectified_divider[:, 0].mean())
    else:
        divider_x = full_width / 2.0

    return rectified_rgb, rectified_mask, transform, inverse_transform, divider_x, divider_found


def project_rectified_point_to_image(point_xy: tuple[float, float], inverse_transform: np.ndarray) -> tuple[float, float]:
    pts = np.array([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(pts, inverse_transform)[0][0]
    return (float(projected[0]), float(projected[1]))


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image.copy()


def render_debug_tile(
    image: np.ndarray,
    title: str,
    subtitle_lines: Iterable[str] = (),
    width: int = 240,
    height: int = 180,
) -> np.ndarray:
    image_rgb = ensure_rgb(image)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    subtitle_lines = list(subtitle_lines)
    text_y = 20
    cv2.putText(canvas, title, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, lineType=cv2.LINE_AA)
    for line in subtitle_lines:
        text_y += 18
        cv2.putText(canvas, line, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, lineType=cv2.LINE_AA)

    top_margin = text_y + 10
    avail_w = width - 12
    avail_h = max(1, height - top_margin - 6)
    img_h, img_w = image_rgb.shape[:2]
    scale = min(avail_w / max(1, img_w), avail_h / max(1, img_h))
    resized_w = max(1, int(round(img_w * scale)))
    resized_h = max(1, int(round(img_h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=interp)
    x0 = (width - resized_w) // 2
    y0 = top_margin + (avail_h - resized_h) // 2
    canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + resized_w, y0 + resized_h), (215, 215, 215), 1, lineType=cv2.LINE_AA)
    return canvas


def make_contact_sheet(panels: list[np.ndarray], columns: int = 2, padding: int = 8, bg_color: tuple[int, int, int] = WHITE) -> np.ndarray:
    if not panels:
        raise ValueError("Cannot make a contact sheet with no panels.")
    panel_h = max(panel.shape[0] for panel in panels)
    panel_w = max(panel.shape[1] for panel in panels)
    rows = int(math.ceil(len(panels) / columns))
    sheet = np.full(
        (rows * panel_h + (rows + 1) * padding, columns * panel_w + (columns + 1) * padding, 3),
        bg_color,
        dtype=np.uint8,
    )
    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        y0 = padding + row * (panel_h + padding)
        x0 = padding + col * (panel_w + padding)
        panel_rgb = ensure_rgb(panel)
        h, w = panel_rgb.shape[:2]
        sheet[y0:y0 + h, x0:x0 + w] = panel_rgb
    return sheet


def draw_rectified_preview(rectified_rgb: np.ndarray, divider_x: float, divider_found: bool) -> np.ndarray:
    preview = rectified_rgb.copy()
    x = int(round(divider_x))
    x = max(0, min(preview.shape[1] - 1, x))
    color = (0, 180, 255) if divider_found else (255, 120, 0)
    cv2.line(preview, (x, 0), (x, preview.shape[0] - 1), color, 2, lineType=cv2.LINE_AA)
    return preview


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask_u8) == 0:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return ((labels == keep).astype(np.uint8) * 255)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask_u8) == 0:
        return mask_u8
    flood = mask_u8.copy()
    h, w = flood.shape[:2]
    floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, floodfill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_u8, holes)


def mask_outline(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    out = image_rgb.copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return out


def analyze_half_crop(half_rgb: np.ndarray, side: str, half_mask: Optional[np.ndarray] = None) -> PipCountDebug:
    def empty_debug(reason: str, empty_mask: np.ndarray) -> PipCountDebug:
        return PipCountDebug(
            count=0,
            confidence=0.0,
            is_valid=False,
            failure_reason=reason,
            raw_blob_count=0,
            layout_score=0.0,
            color_score=0.0,
            color_family=None,
            response_map=empty_mask,
            anchor_mask=empty_mask,
            template_mask=empty_mask,
            candidate_mask=empty_mask,
            accepted_mask=empty_mask,
            rejected_mask=empty_mask,
            overlay_rgb=ensure_rgb(empty_mask),
        )

    if half_rgb is None or half_rgb.size == 0:
        return empty_debug("empty", np.zeros((32, 32), dtype=np.uint8))

    height, width = half_rgb.shape[:2]
    if height < 8 or width < 8:
        return empty_debug("tiny", np.zeros((max(height, 1), max(width, 1)), dtype=np.uint8))

    active_mask = np.any(half_rgb < 250, axis=2).astype(np.uint8) * 255
    active_mask = largest_component(active_mask)
    if cv2.countNonZero(active_mask) < 20:
        return empty_debug("empty-crop", np.zeros((height, width), dtype=np.uint8))

    active_mask = cv2.erode(
        active_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if cv2.countNonZero(active_mask) < 20:
        active_mask = largest_component((np.any(half_rgb < 250, axis=2)).astype(np.uint8) * 255)

    hsv = cv2.cvtColor(half_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(half_rgb, cv2.COLOR_RGB2LAB)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    active_bool = active_mask > 0
    if not np.any(active_bool):
        return empty_debug("empty-crop", np.zeros((height, width), dtype=np.uint8))

    body_selector = active_bool & (sat <= np.percentile(sat[active_bool], 65)) & (val >= np.percentile(val[active_bool], 35))
    if not np.any(body_selector):
        body_selector = active_bool
    bg_color = np.median(lab[body_selector].reshape(-1, 3), axis=0)
    delta = lab.astype(np.float32) - bg_color.reshape(1, 1, 3).astype(np.float32)
    color_distance = np.linalg.norm(delta, axis=2)
    sat_norm = np.clip((sat - 24.0) / 80.0, 0.0, 1.0)
    chroma_norm = np.clip((color_distance - 12.0) / 44.0, 0.0, 1.0)
    response = np.clip(0.55 * sat_norm + 0.45 * chroma_norm, 0.0, 1.0)
    response *= active_bool.astype(np.float32)
    response_blur = cv2.GaussianBlur(response, (3, 3), 0)

    candidate_mask = ((sat > 28.0) & (color_distance > 16.0) & active_bool).astype(np.uint8) * 255
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    candidate_mask = cv2.bitwise_and(candidate_mask, active_mask)

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    active_area = float(cv2.countNonZero(active_mask))
    bbox = _active_bbox(active_mask)
    overlay = mask_outline(half_rgb.copy(), active_mask, (255, 200, 0), 1)
    preliminary_blobs: list[PipBlob] = []
    repair_blobs: list[PipBlob] = []
    rejected_candidates: list[RejectedPipCandidate] = []
    rejected_mask = np.zeros((height, width), dtype=np.uint8)

    for contour in contours:
        reasons: list[str] = []
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = float(cv2.arcLength(contour, True))
        if w <= 0 or h <= 0:
            reasons.append("degenerate")
        area_ratio = area / max(active_area, 1.0)
        if area_ratio < 0.004:
            reasons.append("small")
        if area_ratio > 0.08:
            reasons.append("large")
        aspect_ratio = (w / float(h)) if h > 0 else 99.0
        if aspect_ratio < 0.35 or aspect_ratio > 2.8:
            reasons.append("elongated")
        circularity = 0.0 if perimeter <= 1e-6 else 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.20:
            reasons.append("noncompact")
        extent = area / float(max(1, w * h))
        if extent < 0.15:
            reasons.append("sparse")

        component_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        component_pixels = component_mask > 0
        if int(component_pixels.sum()) <= 0:
            reasons.append("empty")
            mean_sat = 0.0
            mean_chroma = 0.0
            mean_hue = 0.0
        else:
            mean_sat = float(np.mean(sat[component_pixels]))
            mean_chroma = float(np.mean(color_distance[component_pixels]))
            hue_mean = _circular_hue_mean(hue[component_pixels])
            mean_hue = float(hue_mean if hue_mean is not None else 0.0)
        if mean_sat < 43.0:
            reasons.append("low-sat")

        accepted = len(reasons) == 0
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        shape_score = min(1.0, max(0.0, circularity / 0.75))
        area_target = 0.020
        area_score = min(1.0, max(0.0, 1.0 - abs(area_ratio - area_target) / max(area_target, 1e-6)))
        sat_score = min(1.0, max(0.0, (mean_sat - 40.0) / 60.0))
        chroma_score = min(1.0, max(0.0, (mean_chroma - 30.0) / 90.0))
        blob = PipBlob(
            contour=contour,
            component_mask=component_mask,
            center_xy=(float(cx), float(cy)),
            normalized_xy=_normalize_point((float(cx), float(cy)), bbox),
            area_ratio=area_ratio,
            quality=(0.35 * shape_score + 0.25 * area_score + 0.25 * sat_score + 0.15 * chroma_score),
            mean_sat=mean_sat,
            mean_chroma=mean_chroma,
            mean_hue=mean_hue,
        )
        draw_color = (40, 170, 40) if accepted else (220, 80, 80)
        cv2.drawContours(overlay, [contour], -1, draw_color, 2, lineType=cv2.LINE_AA)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), draw_color, 1, lineType=cv2.LINE_AA)
        label = "pip" if accepted else ",".join(reasons[:2])
        cv2.putText(overlay, label, (x, max(12, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, draw_color, 1, lineType=cv2.LINE_AA)
        if accepted:
            preliminary_blobs.append(blob)
        else:
            cv2.drawContours(rejected_mask, [contour], -1, 255, thickness=-1)
            rejected_candidates.append(RejectedPipCandidate(blob=blob, reasons=tuple(reasons)))
            hard_reject = {"degenerate", "empty", "large", "elongated", "sparse"}
            if not (hard_reject & set(reasons)) and 0.002 <= area_ratio <= 0.08 and mean_sat >= 28.0 and mean_chroma >= 24.0:
                repair_blobs.append(
                    replace(
                        blob,
                        quality=float(np.clip(0.62 * blob.quality, 0.0, 0.55)),
                    )
                )

    existing_repair_ids = {id(blob.contour) for blob in repair_blobs}
    for color_repair_blob in _color_matched_repair_blobs(preliminary_blobs, rejected_candidates):
        if id(color_repair_blob.contour) not in existing_repair_ids:
            repair_blobs.append(color_repair_blob)
            existing_repair_ids.add(id(color_repair_blob.contour))

    response_norm = np.clip(response_blur, 0.0, 1.0)
    max_global_response = float(response_norm.max()) if response_norm.size else 0.0
    all_blob_hue = _circular_hue_mean(np.array([blob.mean_hue for blob in preliminary_blobs], dtype=np.float32))
    raw_base_count = len(preliminary_blobs)
    base_count = min(len(preliminary_blobs), 6)
    candidate_results: list[tuple[float, int, float, float, Optional[str], list[PipBlob], str]] = []

    for count, blobs_for_count in _candidate_blob_sets(preliminary_blobs, repair_blobs):
        if count > 6:
            continue
        if count == 0:
            layout_score = float(np.clip(1.0 - max_global_response - 0.22 * len(preliminary_blobs), 0.0, 1.0))
        else:
            layout_score = _relationship_score_for_count(blobs_for_count, count)
        matched_quality = float(np.mean([blob.quality for blob in blobs_for_count])) if blobs_for_count else 0.88
        matched_hue = _circular_hue_mean(np.array([blob.mean_hue for blob in blobs_for_count], dtype=np.float32)) if blobs_for_count else all_blob_hue
        color_score, family = _color_score_for_count(matched_hue, count)
        count_shift = abs(count - base_count)
        edit_penalty = 0.0 if count_shift == 0 else (0.10 if count_shift == 1 else 0.28)
        if count == 6 and color_score > 0.72:
            edit_penalty = max(0.0, edit_penalty - 0.05)
        if count == 4 and len(blobs_for_count) == 4:
            layout_weight = 0.20
            quality_weight = 0.68
            color_weight = 0.12
        elif count == 6 and color_score > 0.72:
            layout_weight = 0.42
            quality_weight = 0.39
            color_weight = 0.19
        else:
            layout_weight = 0.46
            quality_weight = 0.42
            color_weight = 0.12
        combined_score = float(
            np.clip(
                layout_weight * layout_score + quality_weight * matched_quality + color_weight * color_score - edit_penalty,
                0.0,
                1.0,
            )
        )
        blob_ids = {id(blob) for blob in blobs_for_count}
        base_ids = {id(blob) for blob in preliminary_blobs}
        if blob_ids == base_ids:
            source = "base"
        elif len(blobs_for_count) < raw_base_count:
            source = "drop"
        else:
            source = "add"
        candidate_results.append((combined_score, count, layout_score, color_score, family, blobs_for_count, source))

    if not candidate_results:
        candidate_results.append((0.0, 0, 0.0, 0.5, None, [], "empty"))

    candidate_results.sort(key=lambda item: item[0], reverse=True)
    result_by_count = {result[1]: result for result in candidate_results}
    raw_blob_conf = float(np.mean([blob.quality for blob in preliminary_blobs])) if preliminary_blobs else 0.88
    chosen_result = result_by_count.get(base_count, candidate_results[0])
    base_score = chosen_result[0]

    for result in candidate_results:
        score, count, cand_layout_score, _, _, blobs_for_count, source = result
        if count == base_count:
            continue
        margin = score - base_score
        if count == 4:
            continue
        if count == 5:
            required_margin = 0.08 if source == "add" else 0.13
            required_layout = 0.50
        elif count == 6:
            strong_six_color = result[3] > 0.72
            required_margin = 0.02 if source == "add" and strong_six_color else (0.05 if source == "add" else 0.12)
            required_layout = 0.36 if strong_six_color else 0.42
        else:
            required_margin = 0.16
            required_layout = 0.48
        if len(blobs_for_count) == count and margin > required_margin and cand_layout_score > required_layout:
            chosen_result = result
            break

    best_score, final_count, layout_score, color_score, color_family, chosen_blobs, repair_source = chosen_result
    second_score = max((result[0] for result in candidate_results if result[1] != final_count), default=0.0)
    confidence = float(
        np.clip(
            0.12 * best_score + 0.68 * raw_blob_conf + 0.12 * layout_score + 0.08 * color_score,
            0.0,
            1.0,
        )
    )

    chosen_blob_ids = {id(blob) for blob in chosen_blobs}
    base_blob_ids = {id(blob) for blob in preliminary_blobs}
    accepted_mask = np.zeros((height, width), dtype=np.uint8)
    final_rejected_mask = rejected_mask.copy()
    for blob in chosen_blobs:
        cv2.drawContours(accepted_mask, [blob.contour], -1, 255, thickness=-1)
    for blob in preliminary_blobs:
        if id(blob) in chosen_blob_ids:
            cv2.drawContours(overlay, [blob.contour], -1, (30, 175, 30), 2, lineType=cv2.LINE_AA)
        else:
            cv2.drawContours(final_rejected_mask, [blob.contour], -1, 255, thickness=-1)
            cv2.drawContours(overlay, [blob.contour], -1, (230, 170, 0), 2, lineType=cv2.LINE_AA)
            x, y, _, _ = cv2.boundingRect(blob.contour)
            cv2.putText(overlay, "drop", (x, max(12, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 140, 0), 1, lineType=cv2.LINE_AA)
    for blob in chosen_blobs:
        if id(blob) not in base_blob_ids:
            cv2.drawContours(overlay, [blob.contour], -1, (40, 210, 80), 2, lineType=cv2.LINE_AA)
            x, y, _, _ = cv2.boundingRect(blob.contour)
            cv2.putText(overlay, "add", (x, max(12, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (40, 170, 70), 1, lineType=cv2.LINE_AA)

    template_mask = _draw_relationship_mask((height, width), active_mask, chosen_blobs, final_count)
    relation_contours, _ = cv2.findContours((template_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if relation_contours:
        cv2.drawContours(overlay, relation_contours, -1, (0, 200, 255), 1, lineType=cv2.LINE_AA)

    is_valid = True
    failure_reason = None
    if final_count == 0 and base_count == 0 and max_global_response > 0.55:
        is_valid = False
        failure_reason = "zero-conflict"
    elif raw_blob_conf < 0.30 and base_count > 0:
        is_valid = False
        failure_reason = "weak-blobs"
    elif raw_base_count > 6 and not (repair_source == "drop" and final_count == 6 and layout_score > 0.50):
        is_valid = False
        failure_reason = "too-many"
    elif abs(final_count - base_count) >= 2:
        is_valid = False
        failure_reason = "count-jump"
    elif abs(final_count - base_count) == 1 and repair_source != "base" and best_score - base_score < 0.02:
        is_valid = False
        failure_reason = "ambiguous"
    elif final_count > 0 and confidence < 0.22:
        is_valid = False
        failure_reason = "low-confidence"

    cv2.putText(
        overlay,
        f"count={final_count} conf={confidence:.2f}",
        (6, max(14, height - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (20, 20, 20),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"rel={layout_score:.2f} color={color_score:.2f}",
        (6, max(28, height - 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (30, 30, 30),
        1,
        lineType=cv2.LINE_AA,
    )
    if color_family:
        cv2.putText(
            overlay,
            f"family={color_family} blobs={len(preliminary_blobs)} base={base_count} {repair_source}",
            (6, max(42, height - 36)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (45, 45, 45),
            1,
            lineType=cv2.LINE_AA,
        )
    if not is_valid and failure_reason:
        cv2.putText(
            overlay,
            f"gate={failure_reason}",
            (6, max(56, height - 50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 40, 40),
            1,
            lineType=cv2.LINE_AA,
        )

    return PipCountDebug(
        count=final_count,
        confidence=confidence,
        is_valid=is_valid,
        failure_reason=failure_reason,
        raw_blob_count=len(preliminary_blobs),
        layout_score=layout_score,
        color_score=color_score,
        color_family=color_family,
        response_map=(response_norm * 255.0).astype(np.uint8),
        anchor_mask=active_mask,
        template_mask=template_mask,
        candidate_mask=candidate_mask,
        accepted_mask=accepted_mask,
        rejected_mask=final_rejected_mask,
        overlay_rgb=overlay,
    )


def count_pips_in_half_crop(half_rgb: np.ndarray) -> tuple[int, float]:
    cropped = crop_to_nonwhite_bounds(half_rgb)
    if cropped is None or cropped.size == 0:
        return (0, 0.0)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
    bg = np.median(lab.reshape(-1, 3), axis=0)
    delta = lab.astype(np.float32) - bg.reshape(1, 1, 3).astype(np.float32)
    color_distance = np.linalg.norm(delta, axis=2)
    sat = hsv[:, :, 1].astype(np.float32)
    candidate = ((color_distance > 18.0) & (sat > 20.0)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = float(cropped.shape[0] * cropped.shape[1])
    count = 0
    scores = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < area_total * 0.003 or area > area_total * 0.10:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        if aspect < 0.35 or aspect > 2.8:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 1e-6 else 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue
        count += 1
        scores.append(min(1.0, circularity / 0.9))
    confidence = float(np.mean(scores)) if scores else (1.0 if count == 0 else 0.0)
    return (min(count, 6), confidence)


def build_domino_debug_panel(
    rectified_rgb: np.ndarray,
    divider_x: float,
    divider_found: bool,
    face_label: str,
    left_half: np.ndarray,
    right_half: np.ndarray,
    left_debug: Optional[PipCountDebug],
    right_debug: Optional[PipCountDebug],
    belief_result: Optional[DominoBeliefResult] = None,
) -> np.ndarray:
    rectified_preview = draw_rectified_preview(rectified_rgb, divider_x, divider_found)
    rectified_lines = [
        f"divider: {'found' if divider_found else 'missing'}",
        f"face: {face_label}",
    ]
    rectified_tile = render_debug_tile(rectified_preview, "Rectified Domino", rectified_lines)

    if left_debug is None:
        left_tile = render_debug_tile(left_half, "Left Half", ("gated off: no divider",))
        right_tile = render_debug_tile(right_half, "Right Half", ("gated off: no divider",))
        blank_mask = np.full_like(rectified_rgb, 255)
        gate_tile = render_debug_tile(blank_mask, "Blob Masks", ("count skipped", "world map shows '?'"),)
    else:
        left_belief_count = None
        right_belief_count = None
        left_belief_conf = 0.0
        right_belief_conf = 0.0
        if belief_result is not None:
            left_belief_count, right_belief_count = belief_result.counts
            left_belief_conf, right_belief_conf = belief_result.confidences
        left_tile = render_debug_tile(
            left_debug.overlay_rgb,
            "Left Half",
            (
                f"raw {left_debug.count} {left_debug.confidence:.2f}",
                f"belief {left_belief_count if left_belief_count is not None else '?'} {left_belief_conf:.2f}",
                f"layout: {left_debug.layout_score:.2f} color: {left_debug.color_score:.2f}",
                "valid" if left_debug.is_valid else f"gate: {left_debug.failure_reason}",
            ),
        )
        right_tile = render_debug_tile(
            right_debug.overlay_rgb,
            "Right Half",
            (
                f"raw {right_debug.count} {right_debug.confidence:.2f}",
                f"belief {right_belief_count if right_belief_count is not None else '?'} {right_belief_conf:.2f}",
                f"layout: {right_debug.layout_score:.2f} color: {right_debug.color_score:.2f}",
                "valid" if right_debug.is_valid else f"gate: {right_debug.failure_reason}",
            ),
        )
        response_combo = np.concatenate(
            [
                ensure_rgb(left_debug.response_map),
                np.full((left_debug.response_map.shape[0], 6, 3), 255, dtype=np.uint8),
                ensure_rgb(right_debug.response_map),
            ],
            axis=1,
        )
        anchor_combo = np.concatenate(
            [
                ensure_rgb(left_debug.anchor_mask),
                np.full((left_debug.anchor_mask.shape[0], 6, 3), 255, dtype=np.uint8),
                ensure_rgb(right_debug.anchor_mask),
            ],
            axis=1,
        )
        template_combo = np.concatenate(
            [
                ensure_rgb(left_debug.template_mask),
                np.full((left_debug.template_mask.shape[0], 6, 3), 255, dtype=np.uint8),
                ensure_rgb(right_debug.template_mask),
            ],
            axis=1,
        )
        candidate_combo = np.concatenate(
            [
                ensure_rgb(left_debug.candidate_mask),
                np.full((left_debug.candidate_mask.shape[0], 6, 3), 255, dtype=np.uint8),
                ensure_rgb(right_debug.candidate_mask),
            ],
            axis=1,
        )
        accepted_combo = np.concatenate(
            [
                ensure_rgb(left_debug.accepted_mask),
                np.full((left_debug.accepted_mask.shape[0], 6, 3), 255, dtype=np.uint8),
                ensure_rgb(right_debug.accepted_mask),
            ],
            axis=1,
        )
        rejected_combo = np.concatenate(
            [
                ensure_rgb(left_debug.rejected_mask),
                np.full((left_debug.rejected_mask.shape[0], 6, 3), 255, dtype=np.uint8),
                ensure_rgb(right_debug.rejected_mask),
            ],
            axis=1,
        )
        response_tile = render_debug_tile(response_combo, "Response Map", ("color response on the CNN-style half crops",))
        anchor_tile = render_debug_tile(anchor_combo, "Crop Mask", ("non-white half-face region used for blob counting",))
        template_tile = render_debug_tile(template_combo, "Relation Mask", ("relative layout check / repair result",))
        candidate_tile = render_debug_tile(candidate_combo, "Candidate Mask", ("thresholded pixels before contour filtering",))
        rejected_tile = render_debug_tile(rejected_combo, "Rejected Blobs", ("candidates rejected after component scoring",))
        accepted_tile = render_debug_tile(accepted_combo, "Accepted Blobs", ("only accepted pip blobs remain",))
    panels = [rectified_tile, left_tile, right_tile]
    if left_debug is None:
        panels.extend([gate_tile, gate_tile.copy(), gate_tile.copy(), gate_tile.copy(), gate_tile.copy(), gate_tile.copy()])
    else:
        panels.extend([response_tile, anchor_tile, template_tile, candidate_tile, rejected_tile, accepted_tile])
    return make_contact_sheet(panels, columns=3, padding=8)


class ClassicalDominoLabelProvider(DominoLabelProvider):
    BELIEF_WINDOW_SIZE = 8
    MIN_BELIEF_SCORE = 0.35
    MIN_BELIEF_MARGIN = 0.18
    MIN_BELIEF_SHARE = 0.52

    def __init__(self, full_width: int = 120, full_height: int = 60) -> None:
        self.full_width = int(full_width)
        self.full_height = int(full_height)
        self._belief_tracks: list[list[DominoBeliefSample]] = []
        self._belief_tick = 0

    def _track_match_distance(self, observation: DominoObservation) -> float:
        return max(45.0, 0.80 * max(observation.rect_size))

    def _track_for(self, observation: DominoObservation) -> Optional[list[DominoBeliefSample]]:
        if not self._belief_tracks:
            return None
        center = np.array(observation.center_xy, dtype=np.float32)
        max_dist = self._track_match_distance(observation)
        best: Optional[tuple[float, list[DominoBeliefSample]]] = None
        for track in self._belief_tracks:
            if not track:
                continue
            track_center = np.array(track[-1].center_xy, dtype=np.float32)
            dist = float(np.linalg.norm(center - track_center))
            if dist <= max_dist and (best is None or dist < best[0]):
                best = (dist, track)
        return best[1] if best is not None else None

    def _remember_belief_sample(
        self,
        observation: DominoObservation,
        counts: tuple[int, int],
        half_confidences: tuple[float, float],
        face_confidence: float,
    ) -> None:
        center = tuple(float(v) for v in observation.center_xy)
        sample = DominoBeliefSample(
            center_xy=center,
            counts=(int(counts[0]), int(counts[1])),
            half_confidences=(float(half_confidences[0]), float(half_confidences[1])),
            face_confidence=float(face_confidence),
            tick=self._belief_tick,
        )
        track = self._track_for(observation)
        if track is None:
            self._belief_tracks.append([sample])
        else:
            track.append(sample)
            del track[:-self.BELIEF_WINDOW_SIZE]
        self._belief_tracks = [
            track
            for track in self._belief_tracks
            if track and self._belief_tick - track[-1].tick <= self.BELIEF_WINDOW_SIZE * 3
        ]

    def _resolve_half_belief(self, track: list[DominoBeliefSample], half_index: int) -> tuple[Optional[int], float]:
        if not track:
            return (None, 0.0)
        newest_tick = max(sample.tick for sample in track)
        scores: dict[int, float] = {}
        for sample in track[-self.BELIEF_WINDOW_SIZE:]:
            age = max(0, newest_tick - sample.tick)
            recency_weight = max(0.35, 1.0 - (age / max(float(self.BELIEF_WINDOW_SIZE), 1.0)) * 0.65)
            count = int(sample.counts[half_index])
            weight = float(sample.half_confidences[half_index]) * recency_weight
            scores[count] = scores.get(count, 0.0) + weight
        if not scores:
            return (None, 0.0)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_count, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total_score = sum(scores.values())
        belief_share = best_score / max(total_score, 1e-6)
        belief_confidence = float(np.clip(belief_share * min(best_score, 1.0), 0.0, 1.0))
        if best_score < self.MIN_BELIEF_SCORE:
            return (None, belief_confidence)
        if len(ranked) > 1 and best_score - second_score < self.MIN_BELIEF_MARGIN:
            return (None, belief_confidence)
        if belief_share < self.MIN_BELIEF_SHARE:
            return (None, belief_confidence)
        return (best_count, belief_confidence)

    def _resolve_belief(self, observation: DominoObservation) -> DominoBeliefResult:
        track = self._track_for(observation)
        if track is None:
            return DominoBeliefResult(
                counts=(None, None),
                confidences=(0.0, 0.0),
                is_stable=False,
            )
        left_count, left_confidence = self._resolve_half_belief(track, 0)
        right_count, right_confidence = self._resolve_half_belief(track, 1)
        return DominoBeliefResult(
            counts=(left_count, right_count),
            confidences=(left_confidence, right_confidence),
            is_stable=left_count is not None and right_count is not None,
        )

    def annotate(self, image_rgb: np.ndarray, observation: DominoObservation) -> DominoObservation:
        self._belief_tick += 1
        try:
            rectified_result, half_face_result = extract_half_face_crops(
                image_rgb=image_rgb,
                observation=observation,
                full_width=PIPELINE_FULL_WIDTH,
                full_height=PIPELINE_FULL_HEIGHT,
                half_size=DEFAULT_HALF_SIZE,
            )
        except Exception:
            return replace(
                observation,
                face_label="?",
                face_confidence=0.0,
                half_counts=(None, None),
                half_image_centers=None,
                debug_panel=None,
            )

        divider_found = observation.divider_endpoints is not None

        if not divider_found:
            debug_panel = build_domino_debug_panel(
                rectified_rgb=rectified_result.full_crop_rgb,
                divider_x=rectified_result.full_crop_rgb.shape[1] / 2.0,
                divider_found=False,
                face_label="?",
                left_half=rectified_result.full_crop_rgb[:, : rectified_result.full_crop_rgb.shape[1] // 2],
                right_half=rectified_result.full_crop_rgb[:, rectified_result.full_crop_rgb.shape[1] // 2 :],
                left_debug=None,
                right_debug=None,
            )
            return replace(
                observation,
                face_label="?",
                face_confidence=0.0,
                half_counts=(None, None),
                half_image_centers=None,
                debug_panel=debug_panel,
            )

        if half_face_result is None:
            return replace(
                observation,
                face_label="?",
                face_confidence=0.0,
                half_counts=(None, None),
                half_image_centers=None,
                debug_panel=None,
            )

        left_half = half_face_result.left_half_rgb
        right_half = half_face_result.right_half_rgb
        left_debug = analyze_half_crop(left_half, side="left")
        right_debug = analyze_half_crop(right_half, side="right")

        half_image_centers = half_face_result.half_image_centers
        raw_face_valid = left_debug.is_valid and right_debug.is_valid
        raw_face_confidence = float((left_debug.confidence + right_debug.confidence) / 2.0) if raw_face_valid else 0.0
        if raw_face_valid:
            self._remember_belief_sample(
                observation,
                (int(left_debug.count), int(right_debug.count)),
                (float(left_debug.confidence), float(right_debug.confidence)),
                raw_face_confidence,
            )
        belief_result = self._resolve_belief(observation)
        left_count, right_count = belief_result.counts
        face_valid = belief_result.is_stable
        face_label = f"{left_count}-{right_count}" if face_valid else "?"
        face_confidence = (
            float((belief_result.confidences[0] + belief_result.confidences[1]) / 2.0)
            if face_valid
            else 0.0
        )
        debug_panel = build_domino_debug_panel(
            rectified_rgb=rectified_result.full_crop_rgb,
            divider_x=half_face_result.divider_x,
            divider_found=True,
            face_label=face_label,
            left_half=left_half,
            right_half=right_half,
            left_debug=left_debug,
            right_debug=right_debug,
            belief_result=belief_result,
        )

        return replace(
            observation,
            face_label=face_label,
            face_confidence=face_confidence,
            half_counts=(int(left_count), int(right_count)) if face_valid else (None, None),
            half_image_centers=half_image_centers if face_valid else None,
            debug_panel=debug_panel,
        )


class DominoWorldDetector:
    # Reject tiny masks that are usually spurious fragments or noise.
    MIN_MASK_AREA = 140.0
    # The fitted rectangle still needs a plausible domino-sized long dimension in image space.
    MIN_LONG_SIDE_PX = 18.0
    # The short dimension can shrink a lot under foreshortening, but not to zero.
    MIN_SHORT_SIDE_PX = 8.0
    # Flat dominoes can appear nearly square in image space when heavily foreshortened.
    # Keep only very weak image-space shape checks here and let world projection
    # recover the actual placement angle.
    MIN_ASPECT_RATIO = 1.0
    MAX_ASPECT_RATIO = 6.0
    # Require the contour to occupy a reasonable fraction of its min-area rectangle.
    MIN_FILL_RATIO = 0.25
    # Canonical size used when rectifying a detected domino patch for divider analysis.
    RECTIFIED_LONG_PX = 120
    RECTIFIED_SHORT_PX = 60
    # Minimum stripe score before we trust the divider cue instead of falling back.
    DIVIDER_RESPONSE_THRESHOLD = 10.0
    # Smooth the 1D stripe response so single-pixel noise does not dominate the peak.
    DIVIDER_RESPONSE_SIGMA = 2.5
    # Ignore extreme edges of the rectified patch where warp artifacts are common.
    DIVIDER_PEAK_MARGIN_PX = 8
    # Width of the candidate dark divider band in the 1D profile.
    DIVIDER_CENTER_HALF_WIDTH = 2
    # Width of the comparison side bands, which should be lighter than the divider.
    DIVIDER_SIDE_HALF_WIDTH = 4
    # Gap between the dark center band and the lighter side bands.
    DIVIDER_SIDE_GAP = 2
    # Search divider orientation around the coarse rectangle estimate, not all 180 degrees.
    DIVIDER_ANGLE_SEARCH_DEG = 35
    # Angular resolution for the divider sweep in the rectified patch.
    DIVIDER_ANGLE_STEP_DEG = 5
    # The divider should be near the middle of the canonical domino length.
    # Keep this loose for perspective, but reject edge/pip rows far from center.
    DIVIDER_CENTER_MIN_FRACTION = 0.25
    DIVIDER_CENTER_MAX_FRACTION = 0.75
    # In table-camera views, the lower-image side should usually be closer and
    # slightly larger under foreshortening. Require a small margin.
    DIVIDER_NEAR_FAR_AREA_RATIO = 1.4
    # Suppress duplicate detections that cover the same physical domino.
    # Overlap is measured against the smaller mask so partial duplicate masks are removed.
    DUPLICATE_OVERLAP_RATIO = 0.50

    def __init__(
        self,
        conf_threshold: float = 0.3,
        weights_path: Optional[str] = None,
        label_provider: Optional[DominoLabelProvider] = None,
    ) -> None:
        weights_path = resolve_segment_weights(weights_path)

        self.conf_threshold = float(conf_threshold)
        self.weights_path = weights_path
        self.label_provider = label_provider or NullDominoLabelProvider()
        self.model = YOLO(self.weights_path)
        self.model.eval()
        self._kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._last_frame_id: Optional[int] = None
        self._last_observations: list[DominoObservation] = []

    def detect(self, image_rgb: np.ndarray, frame_id: Optional[int] = None) -> list[DominoObservation]:
        """Run segmentation once per frame and cache the structured observations."""
        if image_rgb is None:
            return []
        if frame_id is not None and frame_id == self._last_frame_id:
            return list(self._last_observations)

        fitted = roboflow_fit_resize(image_rgb)
        canvas, scale, pad_x, pad_y, _, _ = fitted
        results = self.model(canvas, conf=self.conf_threshold, verbose=False)
        observations = self._extract_observations(image_rgb, fitted, results[0] if results else None)
        self._last_frame_id = frame_id
        self._last_observations = observations
        return list(observations)

    def latest_observations(self) -> list[DominoObservation]:
        return list(self._last_observations)

    def latest_debug_contact_sheet(self, columns: int = 2) -> Optional[np.ndarray]:
        panels = [obs.debug_panel for obs in self._last_observations if getattr(obs, "debug_panel", None) is not None]
        if not panels:
            return None
        return make_contact_sheet(panels, columns=columns, padding=10)

    def draw_observations(
        self,
        image_rgb: np.ndarray,
        observations: Optional[Iterable[DominoObservation]] = None,
        color: tuple[int, int, int] = (255, 90, 40),
    ) -> np.ndarray:
        """Debug renderer for raw detector output before world-map association."""
        out = image_rgb.copy()
        observations = list(observations if observations is not None else self._last_observations)
        for idx, obs in enumerate(observations):
            quad = np.array(obs.quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [quad], True, color, 2, lineType=cv2.LINE_AA)
            p0 = tuple(int(round(v)) for v in obs.axis_endpoints[0])
            p1 = tuple(int(round(v)) for v in obs.axis_endpoints[1])
            cv2.line(out, p0, p1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cx, cy = obs.center_xy
            label = f"domino {idx}"
            if obs.face_label:
                label += f" {obs.face_label}"
            cv2.putText(
                out,
                label,
                (int(round(cx)) + 4, int(round(cy)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        return out

    def _extract_observations(self, image_rgb: np.ndarray, fitted: tuple, result) -> list[DominoObservation]:
        """Turn one YOLO result into validated domino observations."""
        if result is None:
            return []
        masks = getattr(result, "masks", None)
        if masks is None or len(masks) == 0:
            return []

        boxes = getattr(result, "boxes", None)
        confidences = []
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            confidences = boxes.conf.detach().cpu().numpy().tolist()

        _, scale, pad_x, pad_y, _, _ = fitted
        orig_h, orig_w = image_rgb.shape[:2]
        observations: list[DominoObservation] = []
        for idx, mask_reduced in enumerate(masks):
            confidence = float(confidences[idx]) if idx < len(confidences) else self.conf_threshold
            expanded_mask = map_mask_to_original(mask_reduced.data[0].cpu().numpy(), scale, pad_x, pad_y, orig_w, orig_h)
            observation = self._observation_from_mask(image_rgb, expanded_mask, confidence)
            if observation is None:
                continue
            observations.append(observation)
        observations = self._suppress_duplicate_observations(observations)
        return [self.label_provider.annotate(image_rgb, observation) for observation in observations]

    def _suppress_duplicate_observations(self, observations: list[DominoObservation]) -> list[DominoObservation]:
        """Keep one observation when two masks overlap by more than the duplicate threshold."""
        kept: list[DominoObservation] = []
        for observation in sorted(observations, key=lambda obs: obs.confidence, reverse=True):
            duplicate = False
            for kept_observation in kept:
                if self._mask_overlap_ratio(observation.mask, kept_observation.mask) > self.DUPLICATE_OVERLAP_RATIO:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(observation)
        return kept

    @staticmethod
    def _mask_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        area_a = int(cv2.countNonZero(mask_a))
        area_b = int(cv2.countNonZero(mask_b))
        smaller_area = min(area_a, area_b)
        if smaller_area <= 0:
            return 0.0
        overlap = int(cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b)))
        return float(overlap) / float(smaller_area)

    def _estimate_divider_axis(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        quad: tuple[tuple[float, float], ...],
        center_xy: tuple[float, float],
        long_side: float,
        short_side: float,
        fallback_theta: float,
    ) -> tuple[float, tuple[tuple[float, float], tuple[float, float]], Optional[tuple[tuple[float, float], tuple[float, float]]]]:
        """
        Rectify the domino patch, search for a dark stripe with lighter sides,
        and convert that divider cue back into an image-space long axis.
        """
        def point_segment_distance(pt, a, b):
            ax, ay = a
            bx, by = b
            px, py = pt
            vx = bx - ax
            vy = by - ay
            wx = px - ax
            wy = py - ay
            denom = vx * vx + vy * vy
            if denom <= 1e-6:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
            qx = ax + t * vx
            qy = ay + t * vy
            return math.hypot(px - qx, py - qy)

        def family_axis_from_quad(ordered_quad, divider_pts):
            edges = []
            for i in range(4):
                a = ordered_quad[i]
                b = ordered_quad[(i + 1) % 4]
                edges.append((a, b))

            best_family = None
            best_cost = float("inf")
            for family in ((0, 2), (1, 3)):
                cost = 0.0
                for pt in divider_pts:
                    cost += min(
                        point_segment_distance(pt, *edges[family[0]]),
                        point_segment_distance(pt, *edges[family[1]]),
                    )
                if cost < best_cost:
                    best_cost = cost
                    best_family = family

            if best_family is None:
                return None

            (a0, b0) = edges[best_family[0]]
            (a1, b1) = edges[best_family[1]]
            v0 = np.array([b0[0] - a0[0], b0[1] - a0[1]], dtype=np.float32)
            v1 = np.array([b1[0] - a1[0], b1[1] - a1[1]], dtype=np.float32)
            n0 = float(np.linalg.norm(v0))
            n1 = float(np.linalg.norm(v1))
            if n0 <= 1e-6 or n1 <= 1e-6:
                return None
            v0 /= n0
            v1 /= n1
            if float(v0.dot(v1)) < 0.0:
                v1 = -v1
            avg = v0 + v1
            n = float(np.linalg.norm(avg))
            if n <= 1e-6:
                return None
            avg /= n
            return normalize_axis_angle(math.atan2(float(avg[1]), float(avg[0])))

        def line_box_intersections(nx: float, ny: float, s: float, width: int, height: int):
            pts = []
            eps = 1e-6
            if abs(ny) > eps:
                y = (s - nx * 0.0) / ny
                if 0.0 <= y <= height - 1:
                    pts.append((0.0, float(y)))
                y = (s - nx * (width - 1.0)) / ny
                if 0.0 <= y <= height - 1:
                    pts.append((float(width - 1.0), float(y)))
            if abs(nx) > eps:
                x = (s - ny * 0.0) / nx
                if 0.0 <= x <= width - 1:
                    pts.append((float(x), 0.0))
                x = (s - ny * (height - 1.0)) / nx
                if 0.0 <= x <= width - 1:
                    pts.append((float(x), float(height - 1.0)))
            uniq = []
            for pt in pts:
                if not any((abs(pt[0] - old[0]) < 1e-3 and abs(pt[1] - old[1]) < 1e-3) for old in uniq):
                    uniq.append(pt)
            if len(uniq) < 2:
                return None
            best = (uniq[0], uniq[1])
            best_dist = -1.0
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    dist = (uniq[i][0] - uniq[j][0]) ** 2 + (uniq[i][1] - uniq[j][1]) ** 2
                    if dist > best_dist:
                        best_dist = dist
                        best = (uniq[i], uniq[j])
            return best

        def polygon_area(points: list[tuple[float, float]]) -> float:
            if len(points) < 3:
                return 0.0
            area = 0.0
            for i, (x0, y0) in enumerate(points):
                x1, y1 = points[(i + 1) % len(points)]
                area += x0 * y1 - x1 * y0
            return abs(area) * 0.5

        def polygon_centroid_y(points: list[tuple[float, float]]) -> float:
            if not points:
                return 0.0
            return float(sum(pt[1] for pt in points) / len(points))

        def split_quad_by_line(
            ordered_quad: tuple[tuple[float, float], ...],
            divider_pts: tuple[tuple[float, float], tuple[float, float]],
        ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
            p0 = np.array(divider_pts[0], dtype=np.float32)
            p1 = np.array(divider_pts[1], dtype=np.float32)
            line_vec = p1 - p0

            def signed_distance(pt):
                vec = np.array(pt, dtype=np.float32) - p0
                return float(line_vec[0] * vec[1] - line_vec[1] * vec[0])

            positive = []
            negative = []
            for i, current in enumerate(ordered_quad):
                nxt = ordered_quad[(i + 1) % len(ordered_quad)]
                current_dist = signed_distance(current)
                next_dist = signed_distance(nxt)
                if current_dist >= 0.0:
                    positive.append(current)
                if current_dist <= 0.0:
                    negative.append(current)
                if current_dist * next_dist < 0.0:
                    t = current_dist / (current_dist - next_dist)
                    intersection = (
                        float(current[0] + t * (nxt[0] - current[0])),
                        float(current[1] + t * (nxt[1] - current[1])),
                    )
                    positive.append(intersection)
                    negative.append(intersection)
            return positive, negative

        def near_half_area_is_plausible(ordered_quad, divider_pts) -> bool:
            half_a, half_b = split_quad_by_line(ordered_quad, divider_pts)
            area_a = polygon_area(half_a)
            area_b = polygon_area(half_b)
            if area_a <= 1e-6 or area_b <= 1e-6:
                return False
            near_area, far_area = (area_a, area_b)
            if polygon_centroid_y(half_a) < polygon_centroid_y(half_b):
                near_area, far_area = area_b, area_a
            return near_area >= self.DIVIDER_NEAR_FAR_AREA_RATIO * far_area

        def fallback_axis():
            theta = normalize_axis_angle(fallback_theta)
            dx = math.cos(theta) * long_side / 2.0
            dy = math.sin(theta) * long_side / 2.0
            return theta, ((float(center_xy[0] - dx), float(center_xy[1] - dy)),
                           (float(center_xy[0] + dx), float(center_xy[1] + dy))), None

        def attempt_divider_axis(candidate_theta: float, axis_length: float, require_near_area: bool = False):
            long_theta = normalize_axis_angle(candidate_theta)
            long_dir = np.array([math.cos(long_theta), math.sin(long_theta)], dtype=np.float32)
            short_dir = np.array([-math.sin(long_theta), math.cos(long_theta)], dtype=np.float32)
            center = np.array(center_xy, dtype=np.float32)

            enriched = []
            for pt in quad:
                vec = np.array(pt, dtype=np.float32) - center
                long_coord = float(vec[0] * long_dir[0] + vec[1] * long_dir[1])
                short_coord = float(vec[0] * short_dir[0] + vec[1] * short_dir[1])
                enriched.append((tuple(float(v) for v in pt), long_coord, short_coord))
            enriched.sort(key=lambda item: item[1])
            left = sorted(enriched[:2], key=lambda item: item[2])
            right = sorted(enriched[2:], key=lambda item: item[2])
            if len(left) != 2 or len(right) != 2:
                return None
            src = np.array([left[0][0], right[0][0], right[1][0], left[1][0]], dtype=np.float32)
            ordered_quad = tuple((float(pt[0]), float(pt[1])) for pt in src)
            dst = np.array([
                [0.0, 0.0],
                [self.RECTIFIED_LONG_PX - 1.0, 0.0],
                [self.RECTIFIED_LONG_PX - 1.0, self.RECTIFIED_SHORT_PX - 1.0],
                [0.0, self.RECTIFIED_SHORT_PX - 1.0],
            ], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            Hinv = cv2.getPerspectiveTransform(dst, src)

            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            canon_gray = cv2.warpPerspective(gray, H, (self.RECTIFIED_LONG_PX, self.RECTIFIED_SHORT_PX))
            valid_mask = cv2.warpPerspective(mask, H, (self.RECTIFIED_LONG_PX, self.RECTIFIED_SHORT_PX),
                                             flags=cv2.INTER_NEAREST)
            valid_mask = (valid_mask > 0).astype(np.uint8) * 255
            if cv2.countNonZero(valid_mask) < 20:
                return None

            canon_blur = cv2.GaussianBlur(canon_gray, (5, 5), 0).astype(np.float32)
            yy, xx = np.indices(canon_blur.shape)
            valid = (valid_mask > 0)

            center_hw = self.DIVIDER_CENTER_HALF_WIDTH
            side_hw = self.DIVIDER_SIDE_HALF_WIDTH
            side_gap = self.DIVIDER_SIDE_GAP
            margin = self.DIVIDER_PEAK_MARGIN_PX

            best_idx = None
            best_angle = None
            best_score = -float("inf")
            best_profile = None
            best_min_coord = None
            for angle_deg in range(-self.DIVIDER_ANGLE_SEARCH_DEG, self.DIVIDER_ANGLE_SEARCH_DEG + 1, self.DIVIDER_ANGLE_STEP_DEG):
                alpha = math.radians(angle_deg)
                nx = math.cos(alpha)
                ny = math.sin(alpha)
                coords = xx.astype(np.float32) * nx + yy.astype(np.float32) * ny
                valid_coords = coords[valid]
                if valid_coords.size < 20:
                    continue
                min_coord = float(valid_coords.min())
                bins = np.round(valid_coords - min_coord).astype(np.int32)
                values = canon_blur[valid]
                sums = np.bincount(bins, weights=values)
                counts = np.bincount(bins)
                profile = sums / np.maximum(counts, 1)
                profile = cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (0, 0),
                                           sigmaX=self.DIVIDER_RESPONSE_SIGMA).reshape(-1)
                x_start = margin + side_hw + side_gap + center_hw
                x_stop = len(profile) - (margin + side_hw + side_gap + center_hw)
                if x_stop <= x_start:
                    continue
                for idx in range(x_start, x_stop):
                    center_mean = float(profile[idx - center_hw:idx + center_hw + 1].mean())
                    left_mean = float(profile[idx - side_gap - side_hw:idx - side_gap].mean())
                    right_mean = float(profile[idx + side_gap + 1:idx + side_gap + side_hw + 1].mean())
                    score = min(left_mean - center_mean, right_mean - center_mean)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                        best_angle = alpha
                        best_profile = profile
                        best_min_coord = min_coord

            if best_idx is None or best_score < self.DIVIDER_RESPONSE_THRESHOLD or best_profile is None or best_min_coord is None:
                return None

            center_mean = float(best_profile[best_idx - center_hw:best_idx + center_hw + 1].mean())
            left_mean = float(best_profile[best_idx - side_gap - side_hw:best_idx - side_gap].mean())
            right_mean = float(best_profile[best_idx + side_gap + 1:best_idx + side_gap + side_hw + 1].mean())
            bright_level = min(left_mean, right_mean)
            threshold = center_mean + 0.5 * max(0.0, bright_level - center_mean)

            left_idx = best_idx
            while left_idx > margin and best_profile[left_idx - 1] <= threshold:
                left_idx -= 1
            right_idx = best_idx
            while right_idx + 1 < len(best_profile) - margin and best_profile[right_idx + 1] <= threshold:
                right_idx += 1
            stripe_coord = best_min_coord + (left_idx + right_idx) / 2.0

            nx = math.cos(best_angle)
            ny = math.sin(best_angle)
            center_coord = ((self.RECTIFIED_LONG_PX - 1.0) * 0.5) * nx + ((self.RECTIFIED_SHORT_PX - 1.0) * 0.5) * ny
            long_span = max(1.0, (self.RECTIFIED_LONG_PX - 1.0) * abs(nx) + (self.RECTIFIED_SHORT_PX - 1.0) * abs(ny))
            center_fraction = (stripe_coord - (center_coord - long_span * 0.5)) / long_span
            if (
                center_fraction < self.DIVIDER_CENTER_MIN_FRACTION
                or center_fraction > self.DIVIDER_CENTER_MAX_FRACTION
            ):
                return None
            intersections = line_box_intersections(nx, ny, stripe_coord, self.RECTIFIED_LONG_PX, self.RECTIFIED_SHORT_PX)
            if intersections is None:
                return None
            divider_rectified = np.array([[intersections[0], intersections[1]]], dtype=np.float32)
            divider_image = cv2.perspectiveTransform(divider_rectified, Hinv)[0]
            divider_endpoints = (
                (float(divider_image[0][0]), float(divider_image[0][1])),
                (float(divider_image[1][0]), float(divider_image[1][1])),
            )
            if require_near_area and not near_half_area_is_plausible(ordered_quad, divider_endpoints):
                return None

            axis_theta = family_axis_from_quad(ordered_quad, divider_endpoints)
            if axis_theta is None:
                divider_theta = math.atan2(divider_endpoints[1][1] - divider_endpoints[0][1],
                                           divider_endpoints[1][0] - divider_endpoints[0][0])
                # Fallback: if edge-family disambiguation fails, use the divider as a short-axis cue.
                axis_theta = normalize_axis_angle(divider_theta + math.pi / 2.0)
            dx = math.cos(axis_theta) * axis_length / 2.0
            dy = math.sin(axis_theta) * axis_length / 2.0
            endpoints = ((float(center_xy[0] - dx), float(center_xy[1] - dy)),
                         (float(center_xy[0] + dx), float(center_xy[1] + dy)))
            return best_score, axis_theta, endpoints, divider_endpoints

        # Under strong foreshortening, the apparent minAreaRect long side can be
        # the physical domino short axis. Try both axis assignments and let the
        # divider stripe score choose the better canonical frame.
        attempts = (
            attempt_divider_axis(fallback_theta, long_side),
            attempt_divider_axis(fallback_theta + math.pi / 2.0, short_side, require_near_area=True),
        )
        valid_attempts = [attempt for attempt in attempts if attempt is not None]
        if not valid_attempts:
            return fallback_axis()
        _, axis_theta, endpoints, divider_endpoints = max(valid_attempts, key=lambda item: item[0])
        return axis_theta, endpoints, divider_endpoints

    def _observation_from_mask(self, image_rgb: np.ndarray, mask: np.ndarray, confidence: float) -> Optional[DominoObservation]:
        """Validate one segmentation mask and attach geometry needed by the world map."""
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel5, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        mask_area = float(cv2.contourArea(contour))
        if mask_area < self.MIN_MASK_AREA:
            return None

        # Use a coarse rotated rectangle to estimate the domino footprint and canonical warp.
        rect = cv2.minAreaRect(contour)
        (cx, cy), (dim_a, dim_b), angle_deg = rect
        if dim_a <= 0 or dim_b <= 0:
            return None

        long_side = float(max(dim_a, dim_b))
        short_side = float(min(dim_a, dim_b))
        if long_side < self.MIN_LONG_SIDE_PX or short_side < self.MIN_SHORT_SIDE_PX:
            return None

        aspect_ratio = long_side / max(short_side, 1e-6)
        if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
            return None

        fill_ratio = mask_area / max(dim_a * dim_b, 1.0)
        if fill_ratio < self.MIN_FILL_RATIO:
            return None

        major_angle_deg = float(angle_deg if dim_a >= dim_b else angle_deg + 90.0)
        major_theta = normalize_axis_angle(math.radians(major_angle_deg))
        quad = cv2.boxPoints(rect)
        quad_tuple = tuple((float(pt[0]), float(pt[1])) for pt in quad)
        axis_theta, endpoints, divider_endpoints = self._estimate_divider_axis(
            image_rgb=image_rgb,
            mask=cleaned,
            quad=quad_tuple,
            center_xy=(float(cx), float(cy)),
            long_side=long_side,
            short_side=short_side,
            fallback_theta=float(major_theta),
        )
        return DominoObservation(
            mask=cleaned,
            contour=contour.copy(),
            center_xy=(float(cx), float(cy)),
            rect_size=(long_side, short_side),
            quad=quad_tuple,
            axis_endpoints=endpoints,
            divider_endpoints=divider_endpoints,
            axis_theta=float(axis_theta),
            mask_area=mask_area,
            confidence=float(confidence),
        )

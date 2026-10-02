"""Asynchronous full-frame GPT domino detection and optional geometry fusion.

The local segmenter continues to provide calibrated image geometry.  A
background GPT request reads the complete frame, and its results are matched to
local observations when a local detector is enabled.  If local geometry is
disabled or a GPT tile cannot be matched safely, an approximate GPT-box
observation is retained.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class GPTFallbackObservation:
    center_xy: tuple[float, float]
    rect_size: tuple[float, float]
    quad: tuple[tuple[float, float], ...]
    axis_endpoints: tuple[tuple[float, float], tuple[float, float]]
    divider_endpoints: tuple[tuple[float, float], tuple[float, float]]
    axis_theta: float
    mask_area: float
    confidence: float
    face_label: str
    face_confidence: float
    half_counts: tuple[int, int]
    half_image_centers: tuple[tuple[float, float], tuple[float, float]]
    is_fallen: bool
    observed_frame: Optional[int]
    gpt_geometry_fallback: bool = True


@dataclass(frozen=True)
class _PendingRequest:
    future: Future
    frame_id: Optional[int]
    submitted_at: float
    thumbnail: np.ndarray
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class _GPTSnapshot:
    results: tuple[Any, ...]
    frame_id: Optional[int]
    completed_at: float
    thumbnail: np.ndarray
    image_shape: tuple[int, int]


def _thumbnail(image_rgb: np.ndarray, size: tuple[int, int] = (96, 72)) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _scene_delta(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    return float(np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))))


def _bbox_pixels(result: Any, width: int, height: int) -> tuple[float, float, float, float]:
    left, top, right, bottom = result.bbox_normalized
    x0 = float(left) * width / 1000.0
    y0 = float(top) * height / 1000.0
    x1 = float(right) * width / 1000.0
    y1 = float(bottom) * height / 1000.0
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _observation_bbox(observation: Any) -> tuple[float, float, float, float]:
    quad = np.asarray(observation.quad, dtype=np.float32).reshape(4, 2)
    return (
        float(quad[:, 0].min()),
        float(quad[:, 1].min()),
        float(quad[:, 0].max()),
        float(quad[:, 1].max()),
    )


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ix0 = max(first[0], second[0])
    iy0 = max(first[1], second[1])
    ix1 = min(first[2], second[2])
    iy1 = min(first[3], second[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 1e-6 else 0.0


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_diagonal(box: tuple[float, float, float, float]) -> float:
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _half_centers(
    endpoints: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    first = np.asarray(endpoints[0], dtype=np.float32)
    second = np.asarray(endpoints[1], dtype=np.float32)
    return (
        tuple(float(v) for v in (0.75 * first + 0.25 * second)),
        tuple(float(v) for v in (0.25 * first + 0.75 * second)),
    )


def _gpt_end_centers(
    box: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, y0, x1, y1 = box
    if (x1 - x0) >= (y1 - y0):
        return (((3 * x0 + x1) / 4.0, (y0 + y1) / 2.0),
                ((x0 + 3 * x1) / 4.0, (y0 + y1) / 2.0))
    return (((x0 + x1) / 2.0, (3 * y0 + y1) / 4.0),
            ((x0 + x1) / 2.0, (y0 + 3 * y1) / 4.0))


def _counts_aligned_to_observation(
    observation: Any,
    gpt_result: Any,
    gpt_box: tuple[float, float, float, float],
) -> tuple[int, int]:
    end_a, _ = _gpt_end_centers(gpt_box)
    endpoint_0 = np.asarray(observation.axis_endpoints[0], dtype=np.float32)
    endpoint_1 = np.asarray(observation.axis_endpoints[1], dtype=np.float32)
    end_a_point = np.asarray(end_a, dtype=np.float32)
    values = (int(gpt_result.end_a_pips), int(gpt_result.end_b_pips))
    if np.linalg.norm(endpoint_0 - end_a_point) <= np.linalg.norm(endpoint_1 - end_a_point):
        return values
    return (values[1], values[0])


def _replace_label_fields(
    observation: Any,
    gpt_result: Any,
    gpt_box: tuple[float, float, float, float],
    frame_id: Optional[int],
) -> Any:
    counts = _counts_aligned_to_observation(observation, gpt_result, gpt_box)
    posture = str(gpt_result.posture)
    is_fallen = posture == "fallen"
    confidence = 1.0 if posture != "uncertain" else 0.5
    return replace(
        observation,
        face_label=f"{counts[0]}-{counts[1]}",
        face_confidence=confidence,
        half_counts=counts,
        half_image_centers=_half_centers(observation.axis_endpoints),
        is_fallen=is_fallen,
        observed_frame=frame_id,
    )


def _fallback_observation(
    result: Any,
    box: tuple[float, float, float, float],
    frame_id: Optional[int],
) -> GPTFallbackObservation:
    x0, y0, x1, y1 = box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    if width >= height:
        axis = ((x0, center[1]), (x1, center[1]))
        divider = ((center[0], y0), (center[0], y1))
        theta = 0.0
    else:
        axis = ((center[0], y0), (center[0], y1))
        divider = ((x0, center[1]), (x1, center[1]))
        theta = math.pi / 2.0
    counts = (int(result.end_a_pips), int(result.end_b_pips))
    return GPTFallbackObservation(
        center_xy=center,
        rect_size=(max(width, height), min(width, height)),
        quad=((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        axis_endpoints=axis,
        divider_endpoints=divider,
        axis_theta=theta,
        mask_area=width * height,
        confidence=0.5,
        face_label=f"{counts[0]}-{counts[1]}",
        face_confidence=0.5 if result.posture == "uncertain" else 1.0,
        half_counts=counts,
        half_image_centers=_gpt_end_centers(box),
        is_fallen=str(result.posture) == "fallen",
        observed_frame=frame_id,
    )


def fuse_gpt_with_local_observations(
    local_observations: list[Any],
    gpt_results: tuple[Any, ...] | list[Any],
    image_shape: tuple[int, int],
    frame_id: Optional[int],
) -> list[Any]:
    """Return exactly one labelled observation per readable GPT tile."""
    height, width = image_shape
    readable = [item for item in gpt_results if bool(getattr(item, "readable", False))]
    local_boxes = [_observation_bbox(item) for item in local_observations]
    used_local: set[int] = set()
    fused = []

    for result in readable:
        gpt_box = _bbox_pixels(result, width, height)
        gpt_center = _box_center(gpt_box)
        best: Optional[tuple[float, int]] = None
        for index, local_box in enumerate(local_boxes):
            if index in used_local:
                continue
            local_center = _box_center(local_box)
            distance = math.hypot(
                local_center[0] - gpt_center[0], local_center[1] - gpt_center[1]
            )
            scale = max(_box_diagonal(gpt_box), _box_diagonal(local_box), 1.0)
            normalized_distance = distance / scale
            iou = _bbox_iou(gpt_box, local_box)
            area_ratio = max(_bbox_area(gpt_box), _bbox_area(local_box)) / max(
                min(_bbox_area(gpt_box), _bbox_area(local_box)), 1.0
            )
            if iou < 0.05 and normalized_distance > 0.45:
                continue
            if area_ratio > 2.25:
                continue
            score = 2.0 * iou - normalized_distance
            if best is None or score > best[0]:
                best = (score, index)

        if best is None:
            fused.append(_fallback_observation(result, gpt_box, frame_id))
            continue
        index = best[1]
        used_local.add(index)
        fused.append(
            _replace_label_fields(local_observations[index], result, gpt_box, frame_id)
        )
    return fused


class GPTFullFrameDominoDetector:
    """Detector-compatible wrapper that labels stable frames asynchronously."""

    def __init__(
        self,
        local_detector: Optional[Any],
        client: Any,
        model: str = "gpt-6.1-sol",
        request_interval_seconds: float = 3.0,
        result_ttl_seconds: float = 12.0,
        max_scene_delta: float = 18.0,
        should_request: Optional[Callable[[], bool]] = None,
        recognizer: Optional[Any] = None,
    ) -> None:
        if client is None and recognizer is None:
            raise RuntimeError("GPT domino mode requires OPENAI_API_KEY.")
        if recognizer is None:
            from experiments.domino_gpt.direct_recognizer import DirectDominoGPTRecognizer

            recognizer = DirectDominoGPTRecognizer(
                client=client,
                model=model,
                detail="original",
                reasoning_effort="medium",
            )
        self.local_detector = local_detector
        self.recognizer = recognizer
        self.model = str(model)
        self.request_interval_seconds = float(request_interval_seconds)
        self.result_ttl_seconds = float(result_ttl_seconds)
        self.max_scene_delta = float(max_scene_delta)
        self.should_request = should_request or (lambda: True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="domino-gpt")
        self._pending: Optional[_PendingRequest] = None
        self._snapshot: Optional[_GPTSnapshot] = None
        self._last_submit_time = -float("inf")
        self._latest_observations: list[Any] = []
        self._lock = threading.RLock()

    def _collect_finished_request(self) -> None:
        pending = self._pending
        if pending is None or not pending.future.done():
            return
        self._pending = None
        try:
            results = tuple(pending.future.result())
        except Exception as exc:
            print(f"*** GPT domino recognition failed: {type(exc).__name__}: {exc}")
            return
        self._snapshot = _GPTSnapshot(
            results=results,
            frame_id=pending.frame_id,
            completed_at=time.monotonic(),
            thumbnail=pending.thumbnail,
            image_shape=pending.image_shape,
        )
        elapsed = time.monotonic() - pending.submitted_at
        print(
            f"[GPT domino] frame={pending.frame_id} tiles={len(results)} "
            f"latency={elapsed:.1f}s"
        )

    def _maybe_submit(self, image_rgb: np.ndarray, frame_id: Optional[int]) -> None:
        now = time.monotonic()
        if self._pending is not None:
            return
        if now - self._last_submit_time < self.request_interval_seconds:
            return
        try:
            if not bool(self.should_request()):
                return
        except Exception:
            return
        image_copy = image_rgb.copy()
        pending = _PendingRequest(
            future=self._executor.submit(self.recognizer.recognize, image_copy),
            frame_id=frame_id,
            submitted_at=now,
            thumbnail=_thumbnail(image_copy),
            image_shape=image_copy.shape[:2],
        )
        self._pending = pending
        self._last_submit_time = now
        print(f"[GPT domino] submitted frame={frame_id} model={self.model}")

    def _snapshot_matches(self, image_rgb: np.ndarray) -> bool:
        snapshot = self._snapshot
        if snapshot is None:
            return False
        if time.monotonic() - snapshot.completed_at > self.result_ttl_seconds:
            return False
        if image_rgb.shape[:2] != snapshot.image_shape:
            return False
        return _scene_delta(_thumbnail(image_rgb), snapshot.thumbnail) <= self.max_scene_delta

    def detect(self, image_rgb: np.ndarray, frame_id: Optional[int] = None) -> list[Any]:
        local = (
            self.local_detector.detect(image_rgb, frame_id=frame_id)
            if self.local_detector is not None
            else []
        )
        with self._lock:
            self._collect_finished_request()
            self._maybe_submit(image_rgb, frame_id)
            if self._snapshot_matches(image_rgb):
                snapshot = self._snapshot
                assert snapshot is not None
                observations = fuse_gpt_with_local_observations(
                    local_observations=list(local),
                    gpt_results=snapshot.results,
                    image_shape=image_rgb.shape[:2],
                    frame_id=snapshot.frame_id,
                )
            else:
                observations = list(local)
            self._latest_observations = observations
            return list(observations)

    def latest_observations(self) -> list[Any]:
        with self._lock:
            return list(self._latest_observations)

    def latest_debug_contact_sheet(self, columns: int = 2):
        method = getattr(self.local_detector, "latest_debug_contact_sheet", None)
        return method(columns=columns) if callable(method) else None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

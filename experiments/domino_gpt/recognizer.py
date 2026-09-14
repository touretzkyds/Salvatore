"""GPT pip-count labelling for geometry produced by the domino detectors.

This module deliberately does not detect dominoes or calculate world
coordinates.  It rectifies the quadrilateral and directed axis supplied by a
local detector, asks a stateless OpenAI vision request to count the two ends,
and writes those counts back onto otherwise unchanged observations.

The first crop always maps left-to-right from ``axis_endpoints[0]`` to
``axis_endpoints[1]``.  That contract is important: ``DominoBridge`` uses the
same order to attach pip values to physical ends in the world map.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_DETAIL = "original"
DEFAULT_CROP_SIZE = (512, 256)


class DominoGPTError(RuntimeError):
    """Raised when the vision response cannot safely label every input."""


class _DominoPredictionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domino_id: str
    end_0_pips: Optional[int] = Field(ge=0, le=6)
    end_1_pips: Optional[int] = Field(ge=0, le=6)
    readable: bool
    uncertainty_reason: Optional[str]


class _DominoBatchSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dominoes: list[_DominoPredictionSchema]


@dataclass(frozen=True)
class DominoCrop:
    domino_id: str
    posture: str
    full_crop_rgb: np.ndarray
    end_0_crop_rgb: np.ndarray
    end_1_crop_rgb: np.ndarray


@dataclass(frozen=True)
class DominoPipResult:
    domino_id: str
    end_0_pips: Optional[int]
    end_1_pips: Optional[int]
    readable: bool
    agreement: float
    uncertainty_reason: Optional[str] = None

    @property
    def half_counts(self) -> tuple[Optional[int], Optional[int]]:
        return (self.end_0_pips, self.end_1_pips)


PROMPT = """
You are a visual measurement component for a double-six domino system.
For every supplied domino, count the pips on END 0 and END 1. Valid counts are
integers from 0 through 6. A clear face with no pips is 0; null means the face
cannot be read reliably because it is hidden, blurred, cropped, or not visible.

Each domino is supplied as three images in this exact order: the full rectified
tile, an enlarged END 0 crop, and an enlarged END 1 crop. Never sort or swap the
two counts. Ignore the center divider, borders, glare, shadows, printed labels,
and objects outside the domino face. Return exactly one result for every
domino_id. Set readable=true only when both ends are reliable; otherwise set
readable=false and explain the visual problem briefly.
""".strip()


def _order_quad_along_directed_axis(
    quad: np.ndarray,
    center_xy: Sequence[float],
    axis_endpoints: Sequence[Sequence[float]],
) -> np.ndarray:
    """Order corners so the destination's left side is physical endpoint 0."""
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
    if len(end_0_corners) != 2 or len(end_1_corners) != 2:
        raise ValueError("Could not order the four domino corners.")

    # Destination order: upper-left, upper-right, lower-right, lower-left.
    return np.array(
        [
            end_0_corners[0][0],
            end_1_corners[0][0],
            end_1_corners[1][0],
            end_0_corners[1][0],
        ],
        dtype=np.float32,
    )


def build_domino_crop(
    image_rgb: np.ndarray,
    observation: Any,
    domino_id: str,
    output_size: tuple[int, int] = DEFAULT_CROP_SIZE,
    padding_ratio: float = 0.06,
) -> DominoCrop:
    """Perspective-rectify one local detection without changing end order."""
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot crop an empty camera image.")
    width, height = (int(output_size[0]), int(output_size[1]))
    if width < 2 or height < 2:
        raise ValueError("output_size must be at least 2 by 2 pixels.")

    quad = np.asarray(observation.quad, dtype=np.float32).reshape(4, 2)
    center = np.asarray(observation.center_xy, dtype=np.float32).reshape(2)
    if padding_ratio:
        quad = center + (quad - center) * (1.0 + 2.0 * float(padding_ratio))
    src = _order_quad_along_directed_axis(
        quad=quad,
        center_xy=center,
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
    if split_col < 1 or width - split_col < 1:
        raise ValueError("Rectified domino is too narrow to split.")
    end_0 = full_crop[:, :split_col].copy()
    end_1 = full_crop[:, split_col:].copy()
    posture = "fallen" if bool(getattr(observation, "is_fallen", False)) else "standing"
    return DominoCrop(
        domino_id=str(domino_id),
        posture=posture,
        full_crop_rgb=full_crop,
        end_0_crop_rgb=end_0,
        end_1_crop_rgb=end_1,
    )


def _rgb_png_data_url(image_rgb: np.ndarray) -> str:
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot encode an empty crop.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Domino crops must be RGB images with three channels.")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise DominoGPTError("OpenCV could not encode a domino crop as PNG.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


class DominoGPTRecognizer:
    """Stateless, batched GPT vision classifier for already-detected dominoes."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        detail: str = DEFAULT_DETAIL,
        reasoning_effort: str = "medium",
    ) -> None:
        if detail not in ("low", "high", "original", "auto"):
            raise ValueError("detail must be low, high, original, or auto")
        if reasoning_effort not in ("none", "low", "medium", "high", "xhigh", "max"):
            raise ValueError("Unsupported reasoning_effort")
        self.client = client if client is not None else OpenAI()
        self.model = str(model)
        self.detail = detail
        self.reasoning_effort = reasoning_effort

    def _request_content(self, crops: Sequence[DominoCrop]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Classify {len(crops)} detected domino(es). "
                    "The posture is context only; count the visible face pips."
                ),
            }
        ]
        for crop in crops:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"domino_id={crop.domino_id}; posture={crop.posture}; "
                        "next images are FULL, END 0, END 1."
                    ),
                }
            )
            for image in (
                crop.full_crop_rgb,
                crop.end_0_crop_rgb,
                crop.end_1_crop_rgb,
            ):
                content.append(
                    {
                        "type": "input_image",
                        "image_url": _rgb_png_data_url(image),
                        "detail": self.detail,
                    }
                )
        return content

    def recognize_once(self, crops: Sequence[DominoCrop]) -> dict[str, DominoPipResult]:
        if not crops:
            return {}
        expected_ids = [crop.domino_id for crop in crops]
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("Every DominoCrop must have a unique domino_id.")

        response = self.client.responses.parse(
            model=self.model,
            instructions=PROMPT,
            input=[{"role": "user", "content": self._request_content(crops)}],
            text_format=_DominoBatchSchema,
            reasoning={"effort": self.reasoning_effort},
            store=False,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            output_text = getattr(response, "output_text", "")
            raise DominoGPTError(
                "GPT returned no structured domino result"
                + (f": {output_text}" if output_text else ".")
            )

        items = list(parsed.dominoes)
        returned_ids = [item.domino_id for item in items]
        if len(set(returned_ids)) != len(returned_ids):
            raise DominoGPTError("GPT returned a duplicate domino_id.")
        if set(returned_ids) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(returned_ids))
            extra = sorted(set(returned_ids) - set(expected_ids))
            raise DominoGPTError(
                f"GPT result IDs did not match inputs (missing={missing}, extra={extra})."
            )

        results: dict[str, DominoPipResult] = {}
        for item in items:
            complete = (
                bool(item.readable)
                and item.end_0_pips is not None
                and item.end_1_pips is not None
            )
            results[item.domino_id] = DominoPipResult(
                domino_id=item.domino_id,
                end_0_pips=int(item.end_0_pips) if complete else None,
                end_1_pips=int(item.end_1_pips) if complete else None,
                readable=complete,
                agreement=1.0 if complete else 0.0,
                uncertainty_reason=item.uncertainty_reason,
            )
        return results

    def recognize(
        self,
        crops: Sequence[DominoCrop],
        samples: int = 1,
    ) -> dict[str, DominoPipResult]:
        """Classify a batch, optionally requiring strict multi-call consensus."""
        samples = int(samples)
        if samples < 1:
            raise ValueError("samples must be at least 1")
        if not crops:
            return {}
        runs = [self.recognize_once(crops) for _ in range(samples)]
        if samples == 1:
            return runs[0]

        resolved: dict[str, DominoPipResult] = {}
        required_votes = samples // 2 + 1
        for crop in crops:
            crop_results = [run[crop.domino_id] for run in runs]
            readable_results = [result for result in crop_results if result.readable]
            votes = Counter(result.half_counts for result in readable_results)
            (winning_counts, vote_count) = votes.most_common(1)[0] if votes else ((None, None), 0)
            if vote_count >= required_votes:
                matching = next(
                    result for result in readable_results if result.half_counts == winning_counts
                )
                resolved[crop.domino_id] = DominoPipResult(
                    domino_id=crop.domino_id,
                    end_0_pips=winning_counts[0],
                    end_1_pips=winning_counts[1],
                    readable=True,
                    agreement=vote_count / samples,
                    uncertainty_reason=matching.uncertainty_reason,
                )
            else:
                resolved[crop.domino_id] = DominoPipResult(
                    domino_id=crop.domino_id,
                    end_0_pips=None,
                    end_1_pips=None,
                    readable=False,
                    agreement=vote_count / samples,
                    uncertainty_reason="No strict majority across GPT samples.",
                )
        return resolved


def recognize_observations(
    image_rgb: np.ndarray,
    observations: Iterable[Any],
    recognizer: DominoGPTRecognizer,
    samples: int = 1,
) -> tuple[list[Any], dict[str, DominoPipResult], list[DominoCrop]]:
    """Return copies of detector observations annotated with GPT pip counts."""
    observations = list(observations)
    crops = [
        build_domino_crop(image_rgb, observation, domino_id=f"domino-{index}")
        for index, observation in enumerate(observations)
    ]
    predictions = recognizer.recognize(crops, samples=samples)
    annotated = []
    for index, observation in enumerate(observations):
        domino_id = f"domino-{index}"
        result = predictions[domino_id]
        face_label = (
            f"{result.end_0_pips}-{result.end_1_pips}" if result.readable else "?"
        )
        annotated.append(
            replace(
                observation,
                face_label=face_label,
                face_confidence=result.agreement if result.readable else 0.0,
                half_counts=result.half_counts,
            )
        )
    return annotated, predictions, crops


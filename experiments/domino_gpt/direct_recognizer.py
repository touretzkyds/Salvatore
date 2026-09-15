"""Direct full-frame GPT domino detection and pip recognition.

Unlike ``recognizer.py``, this module does not consume local detector boxes and
does not load any ``.pt`` model. GPT receives one complete camera image and is
responsible for finding each domino, counting its two ends, and estimating its
posture and image-space bounding box.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from experiments.domino_gpt.recognizer import (
    DEFAULT_DETAIL,
    DEFAULT_MODEL,
    DominoGPTError,
    _rgb_png_data_url,
)


class _DirectDominoSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domino_id: str
    end_a_pips: Optional[int] = Field(ge=0, le=6)
    end_b_pips: Optional[int] = Field(ge=0, le=6)
    posture: Literal["standing", "fallen", "uncertain"]
    readable: bool
    bbox_left: int = Field(ge=0, le=1000)
    bbox_top: int = Field(ge=0, le=1000)
    bbox_right: int = Field(ge=0, le=1000)
    bbox_bottom: int = Field(ge=0, le=1000)
    uncertainty_reason: Optional[str]


class _DirectFrameSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dominoes: list[_DirectDominoSchema]


@dataclass(frozen=True)
class DirectDominoResult:
    domino_id: str
    end_a_pips: Optional[int]
    end_b_pips: Optional[int]
    posture: str
    readable: bool
    bbox_normalized: tuple[int, int, int, int]
    uncertainty_reason: Optional[str] = None

    @property
    def face_label(self) -> str:
        if not self.readable:
            return "?"
        return f"{self.end_a_pips}-{self.end_b_pips}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bbox_normalized"] = list(self.bbox_normalized)
        payload["face_label"] = self.face_label
        return payload


DIRECT_PROMPT = """
You are a visual measurement component for a double-six domino system. Inspect
the complete camera image and find every physical domino tile exactly once.
Dominoes are white rectangular game tiles with a black center divider and zero
to six colored pips on each half. Do not mistake monitors, cables, furniture,
reflections, shadows, or printed graphics for dominoes.

For every visible domino:
- Count the pips on its two halves. A clearly blank half is 0. Use null only if
  a half is genuinely hidden, cropped, blurred, or unreadable.
- END A is the visually left half for a mostly horizontal face, or the visually
  top half for a mostly vertical face. END B is the opposite half. Never sort
  the counts numerically.
- posture=standing means the tile is upright on its narrow edge with its face
  roughly vertical. posture=fallen means it lies flat with its face upward.
- Give a tight bounding box around that one tile in normalized image coordinates
  from 0 to 1000: left, top, right, bottom.
- Assign IDs domino-0, domino-1, and so on in left-to-right, then top-to-bottom
  image order.

Return an empty list if there are no dominoes. Set readable=true only when both
pip counts are reliable. Do not merge two adjacent dominoes into one result and
do not report the same physical tile twice.
""".strip()


class DirectDominoGPTRecognizer:
    """Find and classify dominoes directly from one complete RGB image."""

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

    def recognize(self, image_rgb: np.ndarray) -> list[DirectDominoResult]:
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("Cannot recognize dominoes in an empty image.")

        response = self.client.responses.parse(
            model=self.model,
            instructions=DIRECT_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Detect and read every domino in this complete frame.",
                        },
                        {
                            "type": "input_image",
                            "image_url": _rgb_png_data_url(image_rgb),
                            "detail": self.detail,
                        },
                    ],
                }
            ],
            text_format=_DirectFrameSchema,
            reasoning={"effort": self.reasoning_effort},
            store=False,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            output_text = getattr(response, "output_text", "")
            raise DominoGPTError(
                "GPT returned no structured direct-vision result"
                + (f": {output_text}" if output_text else ".")
            )

        items = list(parsed.dominoes)
        ids = [item.domino_id for item in items]
        if len(set(ids)) != len(ids):
            raise DominoGPTError("GPT returned a duplicate domino_id.")

        results = []
        for item in items:
            complete = (
                bool(item.readable)
                and item.end_a_pips is not None
                and item.end_b_pips is not None
            )
            left, right = sorted((int(item.bbox_left), int(item.bbox_right)))
            top, bottom = sorted((int(item.bbox_top), int(item.bbox_bottom)))
            results.append(
                DirectDominoResult(
                    domino_id=item.domino_id,
                    end_a_pips=int(item.end_a_pips) if complete else None,
                    end_b_pips=int(item.end_b_pips) if complete else None,
                    posture=item.posture,
                    readable=complete,
                    bbox_normalized=(left, top, right, bottom),
                    uncertainty_reason=item.uncertainty_reason,
                )
            )
        return results

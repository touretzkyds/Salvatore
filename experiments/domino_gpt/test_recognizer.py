from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.domino_gpt.recognizer import (
    DominoCrop,
    DominoGPTRecognizer,
    build_domino_crop,
    recognize_observations,
)


@dataclass(frozen=True)
class FakeObservation:
    center_xy: tuple[float, float]
    quad: tuple[tuple[float, float], ...]
    axis_endpoints: tuple[tuple[float, float], tuple[float, float]]
    confidence: float = 0.9
    is_fallen: bool = True
    face_label: str | None = None
    face_confidence: float | None = None
    half_counts: tuple[int | None, int | None] | None = None


def _observation(reverse_axis: bool = False) -> FakeObservation:
    endpoints = ((20.0, 50.0), (180.0, 50.0))
    if reverse_axis:
        endpoints = tuple(reversed(endpoints))
    return FakeObservation(
        center_xy=(100.0, 50.0),
        # Intentionally shuffled to ensure the quad's incoming order is irrelevant.
        quad=((180.0, 80.0), (20.0, 20.0), (180.0, 20.0), (20.0, 80.0)),
        axis_endpoints=endpoints,
    )


def _two_color_frame() -> np.ndarray:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    image[20:81, 20:100] = (230, 20, 20)
    image[20:81, 100:181] = (20, 230, 20)
    return image


def _crop(domino_id: str = "domino-0") -> DominoCrop:
    full = np.full((32, 64, 3), 200, dtype=np.uint8)
    return DominoCrop(
        domino_id=domino_id,
        posture="fallen",
        full_crop_rgb=full,
        end_0_crop_rgb=full[:, :32].copy(),
        end_1_crop_rgb=full[:, 32:].copy(),
    )


def _prediction(domino_id: str, end_0: int | None, end_1: int | None, readable=True):
    return SimpleNamespace(
        domino_id=domino_id,
        end_0_pips=end_0,
        end_1_pips=end_1,
        readable=readable,
        uncertainty_reason=None if readable else "blurred",
    )


class FakeResponses:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        batch = self.batches.pop(0)
        return SimpleNamespace(output_parsed=SimpleNamespace(dominoes=batch))


class FakeClient:
    def __init__(self, batches):
        self.responses = FakeResponses(batches)


class DominoGPTTests(unittest.TestCase):
    def test_rectified_crop_preserves_directed_endpoint_order(self):
        image = _two_color_frame()
        crop = build_domino_crop(image, _observation(), "domino-0", padding_ratio=0.0)
        end_0_mean = crop.end_0_crop_rgb[30:-30, 30:-30].mean(axis=(0, 1))
        end_1_mean = crop.end_1_crop_rgb[30:-30, 30:-30].mean(axis=(0, 1))
        self.assertGreater(end_0_mean[0], end_0_mean[1])
        self.assertGreater(end_1_mean[1], end_1_mean[0])

        reversed_crop = build_domino_crop(
            image, _observation(reverse_axis=True), "domino-0", padding_ratio=0.0
        )
        reversed_end_0 = reversed_crop.end_0_crop_rgb[30:-30, 30:-30].mean(axis=(0, 1))
        reversed_end_1 = reversed_crop.end_1_crop_rgb[30:-30, 30:-30].mean(axis=(0, 1))
        self.assertGreater(reversed_end_0[1], reversed_end_0[0])
        self.assertGreater(reversed_end_1[0], reversed_end_1[1])

    def test_recognizer_uses_gpt_56_sol_original_detail_and_structured_parse(self):
        client = FakeClient([[_prediction("domino-0", 2, 5)]])
        recognizer = DominoGPTRecognizer(client=client)
        result = recognizer.recognize_once([_crop()])["domino-0"]

        self.assertEqual(result.half_counts, (2, 5))
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-5.6-sol")
        self.assertIs(call["store"], False)
        self.assertEqual(call["reasoning"], {"effort": "medium"})
        images = [item for item in call["input"][0]["content"] if item["type"] == "input_image"]
        self.assertEqual(len(images), 3)
        self.assertTrue(all(item["detail"] == "original" for item in images))
        self.assertTrue(all(item["image_url"].startswith("data:image/png;base64,") for item in images))

    def test_recognizer_requires_strict_consensus_without_reordering_counts(self):
        client = FakeClient(
            [
                [_prediction("domino-0", 6, 1)],
                [_prediction("domino-0", 6, 1)],
                [_prediction("domino-0", 1, 6)],
            ]
        )
        recognizer = DominoGPTRecognizer(client=client)
        result = recognizer.recognize([_crop()], samples=3)["domino-0"]
        self.assertEqual(result.half_counts, (6, 1))
        self.assertEqual(result.agreement, 2 / 3)

    def test_recognizer_marks_no_majority_as_unknown(self):
        client = FakeClient(
            [
                [_prediction("domino-0", 1, 2)],
                [_prediction("domino-0", 2, 3)],
                [_prediction("domino-0", 3, 4)],
            ]
        )
        recognizer = DominoGPTRecognizer(client=client)
        result = recognizer.recognize([_crop()], samples=3)["domino-0"]
        self.assertFalse(result.readable)
        self.assertEqual(result.half_counts, (None, None))

    def test_recognize_observations_only_replaces_label_fields(self):
        observation = _observation()
        client = FakeClient([[_prediction("domino-0", 3, 0)]])
        recognizer = DominoGPTRecognizer(client=client)
        annotated, predictions, _ = recognize_observations(
            _two_color_frame(), [observation], recognizer
        )
        self.assertEqual(annotated[0].center_xy, observation.center_xy)
        self.assertEqual(annotated[0].quad, observation.quad)
        self.assertEqual(annotated[0].axis_endpoints, observation.axis_endpoints)
        self.assertIs(annotated[0].is_fallen, True)
        self.assertEqual(annotated[0].face_label, "3-0")
        self.assertEqual(annotated[0].half_counts, (3, 0))
        self.assertTrue(predictions["domino-0"].readable)


if __name__ == "__main__":
    unittest.main()

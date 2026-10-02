from __future__ import annotations

from dataclasses import dataclass
import time
import unittest

import numpy as np

from aim_fsm.domino_gpt_detector import (
    GPTFallbackObservation,
    GPTFullFrameDominoDetector,
    fuse_gpt_with_local_observations,
)


@dataclass(frozen=True)
class FakeObservation:
    center_xy: tuple[float, float]
    rect_size: tuple[float, float]
    quad: tuple[tuple[float, float], ...]
    axis_endpoints: tuple[tuple[float, float], tuple[float, float]]
    divider_endpoints: object = None
    axis_theta: float = 0.0
    mask_area: float = 3200.0
    confidence: float = 0.9
    face_label: str | None = None
    face_confidence: float | None = None
    half_counts: tuple[int | None, int | None] | None = None
    half_image_centers: object = None
    debug_panel: object = None
    is_fallen: bool = False
    observed_frame: int | None = None


@dataclass(frozen=True)
class FakeGPTResult:
    end_a_pips: int
    end_b_pips: int
    posture: str
    bbox_normalized: tuple[int, int, int, int]
    readable: bool = True


def _local_observation() -> FakeObservation:
    return FakeObservation(
        center_xy=(100.0, 50.0),
        rect_size=(80.0, 40.0),
        quad=((60.0, 30.0), (140.0, 30.0), (140.0, 70.0), (60.0, 70.0)),
        axis_endpoints=((60.0, 50.0), (140.0, 50.0)),
    )


class FakeLocalDetector:
    def __init__(self, observations):
        self.observations = list(observations)

    def detect(self, image, frame_id=None):
        return list(self.observations)


class FakeRecognizer:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return list(self.results)


class GPTDominoDetectorTests(unittest.TestCase):
    def test_matches_gpt_value_to_local_geometry(self):
        result = FakeGPTResult(2, 5, "standing", (280, 280, 720, 720))
        fused = fuse_gpt_with_local_observations(
            [_local_observation()], [result], image_shape=(100, 200), frame_id=7
        )
        self.assertEqual(len(fused), 1)
        self.assertNotIsInstance(fused[0], GPTFallbackObservation)
        self.assertEqual(fused[0].half_counts, (2, 5))
        self.assertEqual(fused[0].face_label, "2-5")
        self.assertEqual(fused[0].observed_frame, 7)

    def test_uses_gpt_box_when_no_local_geometry_matches(self):
        result = FakeGPTResult(6, 1, "fallen", (700, 100, 950, 500))
        fused = fuse_gpt_with_local_observations(
            [_local_observation()], [result], image_shape=(100, 200), frame_id=9
        )
        self.assertEqual(len(fused), 1)
        self.assertIsInstance(fused[0], GPTFallbackObservation)
        self.assertEqual(fused[0].half_counts, (6, 1))
        self.assertTrue(fused[0].is_fallen)

    def test_background_request_does_not_block_detection(self):
        result = FakeGPTResult(3, 4, "standing", (280, 280, 720, 720))
        recognizer = FakeRecognizer([result])
        detector = GPTFullFrameDominoDetector(
            local_detector=FakeLocalDetector([_local_observation()]),
            client=None,
            recognizer=recognizer,
            request_interval_seconds=100.0,
        )
        image = np.full((100, 200, 3), 180, dtype=np.uint8)
        try:
            first = detector.detect(image, frame_id=1)
            self.assertIsNone(first[0].half_counts)
            for _ in range(50):
                time.sleep(0.01)
                labelled = detector.detect(image, frame_id=2)
                if labelled and labelled[0].half_counts is not None:
                    break
            self.assertEqual(labelled[0].half_counts, (3, 4))
            self.assertEqual(recognizer.calls, 1)
        finally:
            detector.close()

    def test_gpt_only_runs_without_a_local_detector(self):
        result = FakeGPTResult(5, 2, "standing", (100, 200, 900, 800))
        recognizer = FakeRecognizer([result])
        detector = GPTFullFrameDominoDetector(
            local_detector=None,
            client=None,
            recognizer=recognizer,
            request_interval_seconds=100.0,
        )
        image = np.full((100, 200, 3), 180, dtype=np.uint8)
        try:
            self.assertEqual(detector.detect(image, frame_id=10), [])
            labelled = []
            for _ in range(50):
                time.sleep(0.01)
                labelled = detector.detect(image, frame_id=11)
                if labelled:
                    break
            self.assertEqual(len(labelled), 1)
            self.assertIsInstance(labelled[0], GPTFallbackObservation)
            self.assertEqual(labelled[0].half_counts, (5, 2))
            self.assertTrue(labelled[0].gpt_geometry_fallback)
        finally:
            detector.close()


if __name__ == "__main__":
    unittest.main()

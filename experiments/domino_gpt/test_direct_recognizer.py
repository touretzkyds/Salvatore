from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from experiments.domino_gpt.direct_recognizer import DirectDominoGPTRecognizer


def _item(**overrides):
    values = {
        "domino_id": "domino-0",
        "end_a_pips": 2,
        "end_b_pips": 5,
        "posture": "standing",
        "readable": True,
        "bbox_left": 100,
        "bbox_top": 200,
        "bbox_right": 500,
        "bbox_bottom": 600,
        "uncertainty_reason": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Responses:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=SimpleNamespace(dominoes=self.items))


class _Client:
    def __init__(self, items):
        self.responses = _Responses(items)


class DirectRecognizerTests(unittest.TestCase):
    def test_full_frame_request_uses_no_detector_inputs(self):
        client = _Client([_item()])
        recognizer = DirectDominoGPTRecognizer(client=client)
        image = np.full((40, 80, 3), 220, dtype=np.uint8)
        results = recognizer.recognize(image)

        self.assertEqual(results[0].face_label, "2-5")
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "gpt-5.6-sol")
        self.assertEqual(call["reasoning"], {"effort": "medium"})
        content = call["input"][0]["content"]
        images = [part for part in content if part["type"] == "input_image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["detail"], "original")

    def test_unreadable_result_discards_partial_counts(self):
        client = _Client([_item(end_b_pips=None, readable=False)])
        result = DirectDominoGPTRecognizer(client=client).recognize(
            np.ones((10, 10, 3), dtype=np.uint8)
        )[0]
        self.assertFalse(result.readable)
        self.assertEqual(result.face_label, "?")
        self.assertIsNone(result.end_a_pips)

    def test_reversed_box_coordinates_are_normalized(self):
        client = _Client([_item(bbox_left=800, bbox_right=100)])
        result = DirectDominoGPTRecognizer(client=client).recognize(
            np.ones((10, 10, 3), dtype=np.uint8)
        )[0]
        self.assertEqual(result.bbox_normalized, (100, 200, 800, 600))


if __name__ == "__main__":
    unittest.main()

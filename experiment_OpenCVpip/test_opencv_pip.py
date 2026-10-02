from __future__ import annotations

import unittest

import cv2
import numpy as np

from experiment_OpenCVpip.opencv_pip import (
    RectifiedDominoCrop,
    recognize_rectified_crop,
)


def _half(count: int, size: int = 128) -> np.ndarray:
    image = np.full((size, size, 3), 235, dtype=np.uint8)
    layouts = {
        0: [],
        1: [(0.50, 0.50)],
        2: [(0.30, 0.30), (0.70, 0.70)],
        3: [(0.28, 0.28), (0.50, 0.50), (0.72, 0.72)],
        4: [(0.30, 0.30), (0.70, 0.30), (0.30, 0.70), (0.70, 0.70)],
        5: [(0.28, 0.28), (0.72, 0.28), (0.50, 0.50), (0.28, 0.72), (0.72, 0.72)],
        6: [(0.30, 0.20), (0.70, 0.20), (0.30, 0.50), (0.70, 0.50), (0.30, 0.80), (0.70, 0.80)],
    }
    for nx, ny in layouts[count]:
        center = (int(round(nx * (size - 1))), int(round(ny * (size - 1))))
        cv2.circle(image, center, 11, (60, 175, 70), thickness=-1, lineType=cv2.LINE_AA)
    return image


class OpenCVPipExperimentTests(unittest.TestCase):
    def test_counts_two_rectified_halves(self):
        left = _half(2)
        right = _half(5)
        full = np.concatenate([left, right], axis=1)
        crop = RectifiedDominoCrop(full, left, right, split_col=left.shape[1])
        result = recognize_rectified_crop(crop, "domino-0", "standing")
        self.assertTrue(result.readable)
        self.assertEqual((result.end_0_pips, result.end_1_pips), (2, 5))

    def test_low_quality_half_returns_unknown_instead_of_guessing(self):
        left = _half(4)
        right = np.full((128, 128, 3), 255, dtype=np.uint8)
        full = np.concatenate([left, right], axis=1)
        crop = RectifiedDominoCrop(full, left, right, split_col=left.shape[1])
        result = recognize_rectified_crop(crop, "domino-0", "standing")
        self.assertFalse(result.readable)
        self.assertEqual((result.end_0_pips, result.end_1_pips), (None, None))


if __name__ == "__main__":
    unittest.main()

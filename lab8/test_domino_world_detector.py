import math

import cv2
import numpy as np

from lab8.domino_world_detector import (
    ClassicalDominoLabelProvider,
    DominoObservation,
    DominoWorldDetector,
    axis_angle_distance,
    count_pips_in_half_crop,
    normalize_axis_angle,
)


def synthetic_domino_mask(angle_deg=25, center=(120, 90), size=(70, 34), canvas_shape=(180, 240)):
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    rect = (center, size, angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    return mask


def synthetic_divider_domino(angle_deg=0, center=(120, 90), size=(70, 34), divider_axis="horizontal", canvas_shape=(180, 240)):
    mask = synthetic_domino_mask(angle_deg=angle_deg, center=center, size=size, canvas_shape=canvas_shape)
    image = np.full((*canvas_shape, 3), 40, dtype=np.uint8)
    image[mask > 0] = 230
    box = cv2.boxPoints((center, size, angle_deg)).astype(np.float32)
    if divider_axis == "horizontal":
        pts = sorted(box, key=lambda pt: pt[1])
        top = sorted(pts[:2], key=lambda pt: pt[0])
        bottom = sorted(pts[2:], key=lambda pt: pt[0])
        p0 = ((top[0] + bottom[0]) / 2.0).astype(np.int32)
        p1 = ((top[1] + bottom[1]) / 2.0).astype(np.int32)
    else:
        pts = sorted(box, key=lambda pt: pt[0])
        left = sorted(pts[:2], key=lambda pt: pt[1])
        right = sorted(pts[2:], key=lambda pt: pt[1])
        p0 = ((left[0] + right[0]) / 2.0).astype(np.int32)
        p1 = ((left[1] + right[1]) / 2.0).astype(np.int32)
    cv2.line(image, tuple(p0), tuple(p1), (25, 25, 25), 5, lineType=cv2.LINE_AA)
    return image, mask


def synthetic_foreshortened_swapped_divider(canvas_shape=(180, 240)):
    mask = np.zeros(canvas_shape, dtype=np.uint8)
    quad = np.array([[92, 70], [148, 70], [166, 120], [74, 120]], dtype=np.int32)
    cv2.fillConvexPoly(mask, quad, 255)
    image = np.full((*canvas_shape, 3), 40, dtype=np.uint8)
    image[mask > 0] = 230
    cv2.line(image, (89, 86), (151, 86), (25, 25, 25), 5, lineType=cv2.LINE_AA)
    return image, mask


def synthetic_half_crop(count, size=60, color=(70, 180, 70)):
    crop = np.full((size, size, 3), 255, dtype=np.uint8)
    layouts = {
        0: [],
        1: [(0.5, 0.5)],
        2: [(0.32, 0.32), (0.68, 0.68)],
        3: [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)],
        4: [(0.32, 0.32), (0.68, 0.32), (0.32, 0.68), (0.68, 0.68)],
        5: [(0.28, 0.28), (0.72, 0.28), (0.50, 0.50), (0.28, 0.72), (0.72, 0.72)],
        6: [(0.3, 0.20), (0.7, 0.20), (0.3, 0.50), (0.7, 0.50), (0.3, 0.80), (0.7, 0.80)],
    }
    radius = max(4, size // 12)
    for (nx, ny) in layouts[count]:
        center = (int(round(nx * (size - 1))), int(round(ny * (size - 1))))
        cv2.circle(crop, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return crop


def add_vertical_edge_strip(crop, color=(95, 95, 95), width=6, side="right"):
    out = crop.copy()
    if side == "right":
        out[:, -width:] = np.array(color, dtype=np.uint8)
    else:
        out[:, :width] = np.array(color, dtype=np.uint8)
    return out


def add_bottom_edge_strip(crop, color=(95, 95, 95), height=10):
    out = crop.copy()
    out[-height:, :] = np.array(color, dtype=np.uint8)
    return out


def make_test_observation(center=(60.0, 30.0)):
    return DominoObservation(
        mask=np.full((60, 120), 255, dtype=np.uint8),
        contour=np.array([[[0, 0]], [[119, 0]], [[119, 59]], [[0, 59]]], dtype=np.int32),
        center_xy=(float(center[0]), float(center[1])),
        rect_size=(120.0, 60.0),
        quad=((0.0, 0.0), (119.0, 0.0), (119.0, 59.0), (0.0, 59.0)),
        axis_endpoints=((0.0, 29.5), (119.0, 29.5)),
        divider_endpoints=((59.5, 0.0), (59.5, 59.0)),
        axis_theta=0.0,
        mask_area=120.0 * 60.0,
        confidence=0.9,
    )


def remember_test_belief(provider, observation, counts, half_confidences):
    provider._belief_tick += 1
    provider._remember_belief_sample(
        observation,
        counts,
        half_confidences,
        float((half_confidences[0] + half_confidences[1]) / 2.0),
    )


def test_belief_resolver():
    provider = ClassicalDominoLabelProvider(full_width=120, full_height=60)
    observation = make_test_observation()
    for _ in range(3):
        remember_test_belief(provider, observation, (6, 6), (0.82, 0.82))
    remember_test_belief(provider, observation, (4, 5), (0.55, 0.55))
    belief = provider._resolve_belief(observation)
    assert belief.counts == (6, 6), f"repeated 6 beliefs should absorb one flicker, got {belief.counts}"
    assert belief.is_stable, "consistent recent beliefs should be stable"

    for _ in range(4):
        remember_test_belief(provider, observation, (2, 3), (0.92, 0.92))
    belief = provider._resolve_belief(observation)
    assert belief.counts == (2, 3), f"new high-confidence beliefs should replace old counts, got {belief.counts}"

    weak_provider = ClassicalDominoLabelProvider(full_width=120, full_height=60)
    weak_observation = make_test_observation()
    remember_test_belief(weak_provider, weak_observation, (5, 5), (0.20, 0.20))
    weak_belief = weak_provider._resolve_belief(weak_observation)
    assert weak_belief.counts == (None, None), "low-confidence samples should not become stable beliefs"

    separated_provider = ClassicalDominoLabelProvider(full_width=120, full_height=60)
    first = make_test_observation(center=(40.0, 30.0))
    second = make_test_observation(center=(180.0, 30.0))
    remember_test_belief(separated_provider, first, (1, 2), (0.90, 0.90))
    remember_test_belief(separated_provider, second, (5, 6), (0.90, 0.90))
    first_belief = separated_provider._resolve_belief(first)
    second_belief = separated_provider._resolve_belief(second)
    assert first_belief.counts == (1, 2), f"first track should stay separate, got {first_belief.counts}"
    assert second_belief.counts == (5, 6), f"second track should stay separate, got {second_belief.counts}"


def main():
    test_belief_resolver()

    detector = DominoWorldDetector.__new__(DominoWorldDetector)
    detector.MIN_MASK_AREA = 140.0
    detector.MIN_LONG_SIDE_PX = 18.0
    detector.MIN_SHORT_SIDE_PX = 8.0
    detector.MIN_ASPECT_RATIO = 1.0
    detector.MAX_ASPECT_RATIO = 6.0
    detector.MIN_FILL_RATIO = 0.25
    detector._kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detector._kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    detector._kernel9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    mask = synthetic_domino_mask()
    image = np.dstack([mask, mask, mask])
    obs = detector._observation_from_mask(image, mask, 0.9)
    assert obs is not None, "synthetic domino mask should produce an observation"
    assert obs.mask_area > 1000, "mask area should be preserved"
    assert abs(obs.center_xy[0] - 120) < 2 and abs(obs.center_xy[1] - 90) < 2, "center should stay stable"
    assert axis_angle_distance(obs.axis_theta, math.radians(25)) < math.radians(10), "major axis should follow rectangle rotation"
    assert normalize_axis_angle(obs.axis_theta) == obs.axis_theta, "axis angle should be normalized into [0, pi)"

    wrapped = normalize_axis_angle(-0.2)
    assert 0 <= wrapped < math.pi, "negative angles should wrap into axis range"
    assert axis_angle_distance(0.05, math.pi - 0.05) < 0.11, "axis-distance should treat opposite directions as equivalent"

    foreshortened = synthetic_domino_mask(angle_deg=0, center=(120, 90), size=(42, 38))
    foreshortened_image = np.dstack([foreshortened, foreshortened, foreshortened])
    foreshortened_obs = detector._observation_from_mask(foreshortened_image, foreshortened, 0.9)
    assert foreshortened_obs is not None, "foreshortened domino-like masks should not be rejected in image space"

    swapped_image, swapped_mask = synthetic_foreshortened_swapped_divider()
    swapped_obs = detector._observation_from_mask(swapped_image, swapped_mask, 0.9)
    assert swapped_obs is not None, "swapped-axis divider case should produce an observation"
    assert swapped_obs.divider_endpoints is not None, "swapped-axis divider should still be detected"
    assert axis_angle_distance(swapped_obs.axis_theta, math.radians(90)) < math.radians(10), (
        "divider evidence should choose the rotated axis hypothesis when minAreaRect's apparent long side is wrong"
    )

    for pip_count in (0, 2, 5, 6):
        half_crop = synthetic_half_crop(pip_count)
        predicted_count, confidence = count_pips_in_half_crop(half_crop)
        assert predicted_count == pip_count, f"expected {pip_count} pips, got {predicted_count}"
        assert 0.0 <= confidence <= 1.0, "pip-count confidence should stay normalized"

    vertical_strip = add_vertical_edge_strip(synthetic_half_crop(0))
    predicted_count, _ = count_pips_in_half_crop(vertical_strip)
    assert predicted_count == 0, f"vertical edge strip should not count as a pip, got {predicted_count}"

    bottom_strip = add_bottom_edge_strip(synthetic_half_crop(0))
    predicted_count, _ = count_pips_in_half_crop(bottom_strip)
    assert predicted_count == 0, f"bottom edge strip should not count as a pip, got {predicted_count}"

    mixed = add_bottom_edge_strip(add_vertical_edge_strip(synthetic_half_crop(4)))
    predicted_count, _ = count_pips_in_half_crop(mixed)
    assert predicted_count == 4, f"edge strips should not inflate true pip count, got {predicted_count}"

    provider = ClassicalDominoLabelProvider(full_width=120, full_height=60)
    blank = np.full((60, 120, 3), 255, dtype=np.uint8)
    quad = ((0.0, 0.0), (119.0, 0.0), (119.0, 59.0), (0.0, 59.0))
    observation = DominoObservation(
        mask=np.full((60, 120), 255, dtype=np.uint8),
        contour=np.array([[[0, 0]], [[119, 0]], [[119, 59]], [[0, 59]]], dtype=np.int32),
        center_xy=(59.5, 29.5),
        rect_size=(120.0, 60.0),
        quad=quad,
        axis_endpoints=((0.0, 29.5), (119.0, 29.5)),
        divider_endpoints=None,
        axis_theta=0.0,
        mask_area=120.0 * 60.0,
        confidence=0.9,
    )
    annotated = provider.annotate(blank, observation)
    assert annotated.face_label == "?", "missing divider should force unknown face label"
    assert annotated.half_counts == (None, None), "missing divider should suppress half counts"
    assert annotated.debug_panel is not None, "debug visualization should still be available when gating to '?'"
    print("domino world detector geometry checks passed")


if __name__ == "__main__":
    main()

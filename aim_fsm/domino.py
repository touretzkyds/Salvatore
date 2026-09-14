"""Domino perception and dual-model pipeline for aim_fsm using YOLO OBB predictions."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
KNOWN_LENGTH = 4.8     # Domino length in cm
FOCAL_LENGTH = 396.3   # Calibrated focal length in pixels
NUM_CLASSES = 7
MIN_HALF_SIZE = 4
WHITE_THRESHOLD = 250
BORDER_MARGIN = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "weights"))


def resolve_weights(name: str) -> str:
    """Map a bare weights filename onto the repo's weights/ directory.

    Weights are deliberately kept out of git; see weights/README.md.
    """
    if os.path.isabs(name) or os.path.exists(name):
        return name
    return os.path.join(WEIGHTS_DIR, name)

PYTORCH_TENSOR_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def normalize_axis_angle(theta: float | None) -> float | None:
    """Normalizes angle theta to [-pi, pi) for 360-degree orientation support."""
    if theta is None:
        return None
    return float((theta + math.pi) % (2.0 * math.pi) - math.pi)


def directed_long_axis(quad: np.ndarray) -> tuple[float, float]:
    """Return a stable image-space long-axis angle and length for a quad.

    The axis points left-to-right, or top-to-bottom for a vertical domino. This
    gives the GPT crop a repeatable endpoint 0/1 convention without assuming
    that the bottom edge of a standing domino is its long edge.
    """
    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(vectors, axis=1)
    longest_index = int(np.argmax(lengths))
    length = float(lengths[longest_index])
    if length <= 1e-6:
        raise ValueError("Domino quadrilateral has no usable long edge.")
    direction = vectors[longest_index] / length
    if direction[0] < -1e-6 or (abs(float(direction[0])) <= 1e-6 and direction[1] < 0.0):
        direction = -direction
    return (float(math.atan2(direction[1], direction[0])), length)


@dataclass(frozen=True)
class DominoObservation:
    center_xy: tuple[float, float]
    rect_size: tuple[float, float]
    quad: tuple[tuple[float, float], ...]
    axis_endpoints: tuple[tuple[float, float], tuple[float, float]]
    divider_endpoints: Optional[tuple[tuple[float, float], tuple[float, float]]]
    axis_theta: float
    confidence: float
    mask_area: float = 0.0
    distance_cm: float = 0.0
    face_label: Optional[str] = None
    face_confidence: Optional[float] = None
    half_counts: Optional[tuple[Optional[int], Optional[int]]] = None
    is_fallen: bool = False  # Track whether detected as fallen or standing


class CNNDominoLabelProvider:
    def __init__(self, weights_path: str = "fallenhalf.pt") -> None:
        self.device = DEVICE
        weights_path = resolve_weights(weights_path)
        self.weights_path = weights_path
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

        self.transform = PYTORCH_TENSOR_TRANSFORMS

    def preprocess_half(self, half_rgb: np.ndarray) -> Optional[torch.Tensor]:
        try:
            if half_rgb is None or half_rgb.size == 0:
                return None
            if half_rgb.shape[0] < MIN_HALF_SIZE or half_rgb.shape[1] < MIN_HALF_SIZE:
                return None

            gray = cv2.cvtColor(half_rgb, cv2.COLOR_RGB2GRAY)
            filtered = cv2.bilateralFilter(gray, 5, 75, 75)
            equalized = self.clahe.apply(filtered)
            three_channel = np.repeat(equalized[:, :, np.newaxis], 3, axis=2)

            pil_image = Image.fromarray(three_channel)
            tensor = self.transform(pil_image).unsqueeze(0)
            return tensor
        except Exception:
            return None

    def predict_half(self, tensor: Optional[torch.Tensor]) -> tuple[Optional[int], float]:
        if tensor is None or self.model is None:
            return None, 0.0
        try:
            with torch.no_grad():
                tensor = tensor.to(self.device)
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
                pred_class = int(predicted.item())
                conf_val = float(confidence.item())
                if 0 <= pred_class < NUM_CLASSES:
                    return pred_class, conf_val
        except Exception:
            pass
        return None, 0.0


class DominoWorldDetector:
    def __init__(
        self,
        conf_threshold: float = 0.35,
        standing_weights: str = "bestieee.pt",
        fallen_weights: str = "fallen.pt",
        standing_label_weights: str = "different.pt",
        fallen_label_weights: str = "fallenhalf.pt",
        label_mode: str = "cnn",
    ) -> None:
        self.conf_threshold = conf_threshold
        if label_mode not in ("cnn", "none"):
            raise ValueError("label_mode must be 'cnn' or 'none'")
        self.label_mode = label_mode
        
        # Dual main detection models
        self.model_standing = YOLO(resolve_weights(standing_weights))
        self.model_fallen = YOLO(resolve_weights(fallen_weights))
        
        # Keep the original half-face classifiers as the default path.  The
        # GPT experiment uses label_mode="none" so these large models are not
        # loaded and the returned geometry can be labelled in one API batch.
        if self.label_mode == "cnn":
            self.label_standing = CNNDominoLabelProvider(weights_path=standing_label_weights)
            self.label_fallen = CNNDominoLabelProvider(weights_path=fallen_label_weights)
        else:
            self.label_standing = None
            self.label_fallen = None

        self._last_frame_id: Optional[int] = None
        self._last_observations: list[DominoObservation] = []

        self.focal_length = FOCAL_LENGTH
        self.known_length = KNOWN_LENGTH

    def rotate_crop(self, image_rgb: np.ndarray, obb_pts: np.ndarray):
        points = obb_pts.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        center, size, angle = rect
        width, height = size

        if width <= 0 or height <= 0:
            return None, 0.0, rect

        major_length = max(width, height)
        if width < height:
            angle += 90
            width, height = height, width

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image_rgb, M, (image_rgb.shape[1], image_rgb.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )

        x, y = int(center[0]), int(center[1])
        pad = 20
        x1 = max(0, x - int(width / 2) - pad)
        y1 = max(0, y - int(height / 2) - pad)
        x2 = min(image_rgb.shape[1], x + int(width / 2) + pad)
        y2 = min(image_rgb.shape[0], y + int(height / 2) + pad)

        crop = rotated[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, 0.0, rect

        return crop, major_length, rect

    def split_halves(self, image: np.ndarray):
        if image is None or image.size == 0:
            return None, None
        h, w = image.shape[:2]
        mid = w // 2
        if mid < MIN_HALF_SIZE or (w - mid) < MIN_HALF_SIZE:
            return None, None
        return image[:, :mid].copy(), image[:, mid:].copy()

    def remove_white_border(self, image: np.ndarray, threshold=WHITE_THRESHOLD, margin=BORDER_MARGIN):
        if image is None or image.size == 0:
            return image
        keep_mask = np.any(image < threshold, axis=2)
        if not np.any(keep_mask):
            return image
        ys, xs = np.where(keep_mask)
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(image.shape[0], int(ys.max()) + margin + 1)
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(image.shape[1], int(xs.max()) + margin + 1)
        trimmed = image[y0:y1, x0:x1]
        return trimmed if (trimmed is not None and trimmed.size > 0) else image

    def _extract_quads(self, results, is_fallen: bool):
        if not results:
            return []
        result = results[0]
        obb = getattr(result, "obb", None)
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        quads_info = []

        if obb is not None and len(obb) > 0:
            raw_quads = list(obb.xyxyxyxy.cpu().numpy())
            confs = obb.conf.cpu().numpy().tolist() if obb.conf is not None else [self.conf_threshold] * len(raw_quads)
            for q, c in zip(raw_quads, confs):
                quads_info.append((q, None, float(c), is_fallen))
        elif masks is not None and len(masks) > 0:
            mask_polys = masks.xy
            confs = boxes.conf.cpu().numpy().tolist() if boxes is not None and boxes.conf is not None else [self.conf_threshold] * len(mask_polys)
            for poly, c in zip(mask_polys, confs):
                if len(poly) >= 4:
                    rect = cv2.minAreaRect(np.array(poly, dtype=np.float32))
                    box = cv2.boxPoints(rect)
                    quads_info.append((box, None, float(c), is_fallen))
        elif boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [self.conf_threshold] * len(xyxy)
            for box, c in zip(xyxy, confs):
                x1b, y1b, x2b, y2b = box
                quad = np.array([[x1b, y1b], [x2b, y1b], [x2b, y2b], [x1b, y2b]], dtype=np.float32)
                quads_info.append((quad, max(x2b - x1b, y2b - y1b), float(c), is_fallen))

        return quads_info

    def detect(self, image_rgb: np.ndarray, frame_id: Optional[int] = None) -> list[DominoObservation]:
        if image_rgb is None:
            return []
        if frame_id is not None and frame_id == self._last_frame_id:
            return list(self._last_observations)

        # Run standing model
        try:
            res_standing = self.model_standing(image_rgb, conf=self.conf_threshold, verbose=False)
        except Exception:
            res_standing = None

        # Run fallen model
        try:
            res_fallen = self.model_fallen(image_rgb, conf=self.conf_threshold, verbose=False)
        except Exception:
            res_fallen = None

        quads_standing = self._extract_quads(res_standing, is_fallen=False)
        quads_fallen = self._extract_quads(res_fallen, is_fallen=True)

        all_quads = quads_standing + quads_fallen
        if not all_quads:
            self._last_frame_id = frame_id
            self._last_observations = []
            return []

        # Non-Maximum Suppression / High Confidence Preference
        # Suppress overlapping predictions from both models based on highest confidence
        filtered_quads = []
        all_quads.sort(key=lambda item: item[2], reverse=True)  # Sort by confidence descending

        for item in all_quads:
            raw_obb, fallback_len, conf, is_fallen = item
            c_x, c_y = np.mean(raw_obb[:, 0]), np.mean(raw_obb[:, 1])
            
            overlap = False
            for prev in filtered_quads:
                p_obb = prev[0]
                pc_x, pc_y = np.mean(p_obb[:, 0]), np.mean(p_obb[:, 1])
                if math.hypot(c_x - pc_x, c_y - pc_y) < 30.0:  # Distance threshold for overlap
                    overlap = True
                    break
            if not overlap:
                filtered_quads.append(item)

        observations: list[DominoObservation] = []

        for raw_obb, fallback_length, conf, is_fallen in filtered_quads:
            quad_pts = np.array(raw_obb, dtype=np.float32)
            try:
                stable_axis_theta, stable_pixel_length = directed_long_axis(quad_pts)
            except ValueError:
                continue
            pixel_length = stable_pixel_length

            crop, crop_major_len, rect = self.rotate_crop(image_rgb, raw_obb)
            if pixel_length <= 0:
                pixel_length = crop_major_len if crop_major_len > 0 else (fallback_length or 0.0)

            if crop is None or crop.size == 0 or pixel_length <= 0:
                continue

            h, w, _ = crop.shape
            if h < 20 or w < 20:
                continue

            left_pred: Optional[int] = None
            right_pred: Optional[int] = None
            face_label: Optional[str] = None
            face_confidence: Optional[float] = None
            if self.label_mode == "cnn":
                left_raw, right_raw = self.split_halves(crop)
                if left_raw is None or right_raw is None:
                    continue

                left_clean = self.remove_white_border(left_raw)
                right_clean = self.remove_white_border(right_raw)

                # Choose half-face classifier depending on best detection model.
                active_label_provider = self.label_fallen if is_fallen else self.label_standing
                if active_label_provider is None:
                    continue

                left_tensor = active_label_provider.preprocess_half(left_clean)
                right_tensor = active_label_provider.preprocess_half(right_clean)

                left_pred, left_conf = active_label_provider.predict_half(left_tensor)
                right_pred, right_conf = active_label_provider.predict_half(right_tensor)

                if left_pred is None or right_pred is None:
                    continue

                face_label = f"{left_pred}-{right_pred}"
                face_confidence = float(min(left_conf, right_conf))
            distance_cm = float((self.known_length * self.focal_length) / pixel_length)

            (cx, cy), (dim_a, dim_b), angle_deg = rect
            mask_area = float(dim_a * dim_b)

            if self.label_mode == "none":
                # The experiment needs a directed *long* axis so GPT's END 0
                # and END 1 map back to the same physical world-map ends.
                major_theta = stable_axis_theta
            else:
                # Preserve the original CNN pipeline's orientation behavior.
                sorted_by_y = sorted(quad_pts, key=lambda pt: pt[1], reverse=True)
                p_base1, p_base2 = sorted_by_y[0], sorted_by_y[1]
                if p_base1[0] > p_base2[0]:
                    p_base1, p_base2 = p_base2, p_base1
                dx_base = p_base2[0] - p_base1[0]
                dy_base = p_base2[1] - p_base1[1]
                major_theta = float(normalize_axis_angle(math.atan2(dy_base, dx_base)))

            dx = math.cos(major_theta) * (pixel_length / 2.0)
            dy = math.sin(major_theta) * (pixel_length / 2.0)
            axis_endpoints = ((float(cx - dx), float(cy - dy)), (float(cx + dx), float(cy + dy)))

            quad_tuple = tuple((float(pt[0]), float(pt[1])) for pt in raw_obb)

            obs = DominoObservation(
                center_xy=(float(cx), float(cy)),
                rect_size=(float(max(dim_a, dim_b)), float(min(dim_a, dim_b))),
                quad=quad_tuple,
                axis_endpoints=axis_endpoints,
                divider_endpoints=None,
                axis_theta=major_theta,
                confidence=conf,
                mask_area=mask_area,
                distance_cm=distance_cm,
                face_label=face_label,
                face_confidence=face_confidence,
                half_counts=(left_pred, right_pred) if self.label_mode == "cnn" else (None, None),
                is_fallen=is_fallen,
            )
            observations.append(obs)

        self._last_frame_id = frame_id
        self._last_observations = observations
        return list(observations)

    def latest_observations(self) -> list[DominoObservation]:
        return list(self._last_observations)

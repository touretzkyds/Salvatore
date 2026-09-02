"""
domino_node.py
==============
VEX AIM FSM State Machine for Autonomous Domino Perception,
Fallen Domino Path Tracking, & World Mapping.

Execution in FSM Console:
  runfsm("domino_node")
  gui()
"""

import math
import time
import cv2
import numpy as np
from typing import List, Tuple, Optional

from ultralytics import YOLO

import aim_fsm
from aim_fsm.nodes import StateNode
from aim_fsm.program import StateMachineProgram

# Safe imports for Pose across aim_fsm versions
try:
    from aim_fsm.pose import Pose
except ImportError:
    try:
        from aim_fsm import Pose
    except ImportError:
        class Pose:
            def __init__(self, x=0.0, y=0.0, z=0.0, theta=0.0):
                self.x = x
                self.y = y
                self.z = z
                self.theta = theta


# ==============================================================================
# 1. PERCEPTION DATA STRUCTURES & DETECTOR
# ==============================================================================

class DominoObservation:
    """Dataclass storing frame observations across all 3 YOLO models."""
    def __init__(
        self,
        quad: List[Tuple[float, float]],
        center_xy: Tuple[float, float],
        axis_endpoints: Tuple[Tuple[float, float], Tuple[float, float]],
        divider_endpoints: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
        half_counts: Tuple[Optional[int], Optional[int]],
        mask_area: float,
        confidence: float,
        face_confidence: float,
        is_fallen: bool = False
    ):
        self.quad = quad
        self.center_xy = center_xy
        self.axis_endpoints = axis_endpoints
        self.divider_endpoints = divider_endpoints
        self.half_counts = half_counts
        self.mask_area = mask_area
        self.confidence = confidence
        self.face_confidence = face_confidence
        self.is_fallen = is_fallen

    @property
    def face_label(self) -> Optional[str]:
        if self.is_fallen:
            return "FALLEN"
        a, b = self.half_counts
        if a is None or b is None:
            return None
        return f"{a}|{b}"


class DominoDetector:
    """Multi-model pipeline: bestieee.pt, different.pt, and fallen.pt."""
    def __init__(
        self,
        seg_model_path: str = "bestieee.pt",
        cls_model_path: str = "different.pt",
        fallen_model_path: str = "fallen.pt",
        conf_threshold: float = 0.50,
    ):
        self.conf_threshold = conf_threshold
        
        print(f"[DominoNode] Loading segmentation model: {seg_model_path}")
        self.seg_model = YOLO(seg_model_path)
        
        print(f"[DominoNode] Loading classification model: {cls_model_path}")
        self.cls_model = YOLO(cls_model_path)
        
        print(f"[DominoNode] Loading fallen domino model: {fallen_model_path}")
        self.fallen_model = YOLO(fallen_model_path)

    def detect(self, image: np.ndarray) -> List[DominoObservation]:
        if image is None or image.size == 0:
            return []

        observations = []
        
        # 1. Run fallen domino / path segmentation first
        fallen_masks_list = []
        try:
            fallen_results = self.fallen_model.predict(source=image, conf=self.conf_threshold, verbose=False)
            if fallen_results and len(fallen_results[0]) > 0 and fallen_results[0].masks is not None:
                for f_mask in fallen_results[0].masks.data:
                    f_np = (f_mask.cpu().numpy() * 255).astype(np.uint8)
                    if f_np.shape[:2] != image.shape[:2]:
                        f_np = cv2.resize(f_np, (image.shape[1], image.shape[0]))
                    fallen_masks_list.append(f_np)
        except Exception as e:
            pass

        # 2. Run standard domino segmentation
        seg_results = self.seg_model.predict(source=image, conf=self.conf_threshold, verbose=False)

        if not seg_results or len(seg_results[0]) == 0 or seg_results[0].masks is None:
            return []

        res = seg_results[0]
        masks = res.masks

        for idx, mask in enumerate(masks.data):
            conf = float(res.boxes.conf[idx].cpu().numpy())
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)

            if mask_np.shape[:2] != image.shape[:2]:
                mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            if area < 150:
                continue

            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            quad = [(float(pt[0]), float(pt[1])) for pt in box]

            cx, cy = float(rect[0][0]), float(rect[0][1])
            (w, h) = rect[1]
            angle = rect[2]

            if w < h:
                w, h = h, w
                angle += 90.0

            angle_rad = math.radians(angle)
            dx = (w / 2.0) * math.cos(angle_rad)
            dy = (w / 2.0) * math.sin(angle_rad)

            axis0 = (cx - dx, cy - dy)
            axis1 = (cx + dx, cy + dy)

            div_dx = (h / 2.0) * math.cos(angle_rad + math.pi / 2.0)
            div_dy = (h / 2.0) * math.sin(angle_rad + math.pi / 2.0)
            divider0 = (cx - div_dx, cy - div_dy)
            divider1 = (cx + div_dx, cy + div_dy)

            # Check if this detection overlaps with fallen domino mask
            is_fallen = False
            for f_mask in fallen_masks_list:
                overlap = cv2.bitwise_and(mask_np, f_mask)
                if cv2.countNonZero(overlap) > (area * 0.3):
                    is_fallen = True
                    break

            val1, val2 = None, None
            face_conf = 0.0

            # Only do pip classification if domino is standing/upright
            if not is_fallen:
                half1_center = (cx - (w / 4.0) * math.cos(angle_rad), cy - (w / 4.0) * math.sin(angle_rad))
                half2_center = (cx + (w / 4.0) * math.cos(angle_rad), cy + (w / 4.0) * math.sin(angle_rad))

                crop_size = int(max(w / 2.0, h))
                crop1 = self._crop_half(image, half1_center, crop_size)
                crop2 = self._crop_half(image, half2_center, crop_size)

                val1, conf1 = self._classify_half(crop1) if crop1.size > 0 else (None, 0.0)
                val2, conf2 = self._classify_half(crop2) if crop2.size > 0 else (None, 0.0)
                face_conf = (conf1 + conf2) / 2.0 if (conf1 and conf2) else 0.0

            obs = DominoObservation(
                quad=quad,
                center_xy=(cx, cy),
                axis_endpoints=(axis0, axis1),
                divider_endpoints=(divider0, divider1),
                half_counts=(val1, val2),
                mask_area=area,
                confidence=conf,
                face_confidence=face_conf,
                is_fallen=is_fallen
            )
            observations.append(obs)

        return observations

    def _crop_half(self, image: np.ndarray, center: Tuple[float, float], size: int) -> np.ndarray:
        cx, cy = int(round(center[0])), int(round(center[1]))
        r = max(size // 2, 12)
        h, w = image.shape[:2]
        x1, x2 = max(0, cx - r), min(w, cx + r)
        y1, y2 = max(0, cy - r), min(h, cy + r)
        return image[y1:y2, x1:x2]

    def _classify_half(self, crop: np.ndarray) -> Tuple[Optional[int], float]:
        try:
            results = self.cls_model.predict(source=crop, verbose=False)
            if results and len(results[0]) > 0:
                top_class = int(results[0].probs.top1)
                top_conf = float(results[0].probs.top1conf.cpu().numpy())
                return top_class, top_conf
        except Exception:
            pass
        return None, 0.0


# ==============================================================================
# 2. VEX AIM FSM STATE MACHINE PROGRAM & NODE
# ==============================================================================

class DominoTrackerNode(StateNode):
    """Core tracking node executing inference and feeding WorldMap."""

    def __init__(self):
        super().__init__()
        self.detector: Optional[DominoDetector] = None
        self.latest_obs: List[DominoObservation] = []
        self.frame_count = 0

    def start(self, event=None):
        super().start(event)
        print("[DominoTrackerNode] Tracker node initialized with fallen.pt support...")

        if not hasattr(self.robot, "domino_detector") or self.robot.domino_detector is None:
            self.detector = DominoDetector(
                seg_model_path="bestieee.pt",
                cls_model_path="different.pt",
                fallen_model_path="fallen.pt",
                conf_threshold=0.50
            )
            self.robot.domino_detector = self.detector
        else:
            self.detector = self.robot.domino_detector

    def user_image(self, event):
        """Triggered automatically when camera frames stream through FSM."""
        image = getattr(event, "image", None)
        if self.detector is None or image is None:
            return

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print(f"[DominoTrackerNode] Frame #{self.frame_count} processed...")

        self.latest_obs = self.detector.detect(image)

        if self.latest_obs:
            fallen_cnt = sum(1 for o in self.latest_obs if o.is_fallen)
            print(f"[DominoTrackerNode] Detected {len(self.latest_obs)} dominoes ({fallen_cnt} fallen).")

        if hasattr(self.robot, "worldmap") and self.robot.worldmap is not None:
            self._update_worldmap(self.latest_obs)

    def _update_worldmap(self, observations: List[DominoObservation]):
        """Projects observations to 3D coordinates and updates worldmap."""
        wm = self.robot.worldmap

        for i, obs in enumerate(observations):
            world_x = (obs.center_xy[0] - 320.0) * 0.8
            world_y = (240.0 - obs.center_xy[1]) * 0.8

            p1, p2 = obs.axis_endpoints
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            theta = math.atan2(dy, dx)

            val_a = obs.half_counts[0] if obs.half_counts[0] is not None else 0
            val_b = obs.half_counts[1] if obs.half_counts[1] is not None else 0

            domino_id = f"domino_{i}"
            pose = Pose(x=world_x, y=world_y, z=1.0 if obs.is_fallen else 4.0, theta=theta)

            if hasattr(wm, "update_domino"):
                wm.update_domino(
                    domino_id=domino_id,
                    pose=pose,
                    value_a=val_a,
                    value_b=val_b,
                    confidence=obs.confidence,
                    is_fallen=obs.is_fallen
                )
            elif hasattr(wm, "add_object"):
                wm.add_object(domino_id, pose)
            elif hasattr(wm, "objects"):
                wm.objects[domino_id] = pose

    def user_annotate(self, image: np.ndarray) -> np.ndarray:
        """Annotates live camera display with red (fallen) or green (upright) bounds."""
        annotated = image.copy()

        for obs in self.latest_obs:
            pts = np.array(obs.quad, dtype=np.int32).reshape((-1, 1, 2))
            
            # Red outline for fallen domino, Green for active domino
            color = (0, 0, 255) if obs.is_fallen else (0, 255, 0)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            if obs.divider_endpoints and not obs.is_fallen:
                p1 = (int(obs.divider_endpoints[0][0]), int(obs.divider_endpoints[0][1]))
                p2 = (int(obs.divider_endpoints[1][0]), int(obs.divider_endpoints[1][1]))
                cv2.line(annotated, p1, p2, (255, 0, 0), 2)

            label = obs.face_label or "?|?"
            cx, cy = int(obs.center_xy[0]), int(obs.center_xy[1])
            cv2.putText(
                annotated,
                label,
                (cx - 25, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        return annotated


class domino_node(StateMachineProgram):
    """Entry state machine class bound for CLI runfsm execution."""

    def __init__(self):
        super().__init__()

    def setup(self):
        self.tracker = DominoTrackerNode()
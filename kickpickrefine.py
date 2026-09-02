import os
import cv2
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

from aim_fsm import *
from ultralytics import YOLO

# ==========================================
# 1. SETUP & CALIBRATION CONSTANTS
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_LENGTH = 4.8      # Real domino length in cm
FOCAL_LENGTH = 396.3    # Ground-truth focal value
CAMERA_HFOV_DEG = 62.0  # Horizontal angle ref

# Model Paths
SEG_MODEL_PATH = os.path.join(PROJECT_DIR, "bestieee.pt")
CLS_MODEL_PATH = os.path.join(PROJECT_DIR, "different.pt")
FALLEN_MODEL_PRIMARY = os.path.join(PROJECT_DIR, "fallen.pt")
FALLEN_MODEL_FALLBACK = os.path.join(
    PROJECT_DIR, "runs_play_dominos", "play_dominos_seg_v1_img160", "weights", "bestieee.pt"
)

# Classifier / Preprocessing Params
NUM_CLASSES = 7
MIN_HALF_SIZE = 4
WHITE_THRESHOLD = 250
BORDER_MARGIN = 4

# Fallen Detection Params
SETTLE_DELAY_SEC = 0.20
MODEL_SIZE = 160
PREDICT_CONF = 0.05
PREDICT_IOU = 0.45
DETECT_TRIES = 5
DETECT_RETRY_DELAY_SEC = 0.15

# Motion Parameters
ALIGN_TOL_PX_NEAR = 4.0
ALIGN_TOL_PX_FAR = 10.0
NEAR_ALIGN_DISTANCE_CM = 18.0
MIN_TURN_DEG = 2.0
MAX_TURN_DEG = 10.0
MAX_TURN_STEPS = 6

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PYTORCH_TENSOR_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================
# 2. HELPER UTILITIES
# ==========================================
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def sign(value):
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def wrap_deg(deg):
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def estimate_angle_and_distance_cm(center_x: float, image_w: float, major_length_px: float):
    focal_px_for_angle = (image_w * 0.5) / math.tan(math.radians(CAMERA_HFOV_DEG * 0.5))
    angle_deg = math.degrees(math.atan2(center_x - (image_w * 0.5), focal_px_for_angle))
    distance_cm = (KNOWN_LENGTH * FOCAL_LENGTH) / max(major_length_px, 1.0)
    return angle_deg, distance_cm


def estimate_divider_x(image_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float):
    h, w = image_bgr.shape[:2]
    ix1 = max(0, min(int(round(x1)), w - 1))
    iy1 = max(0, min(int(round(y1)), h - 1))
    ix2 = max(ix1 + 1, min(int(round(x2)), w))
    iy2 = max(iy1 + 1, min(int(round(y2)), h))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    roi = image_bgr[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rw = gray.shape[1]
    if rw < 12:
        return None

    c0 = int(0.30 * rw)
    c1 = int(0.70 * rw)
    if c1 <= c0 + 1:
        return None

    profile = gray.mean(axis=0)
    band = profile[c0:c1]
    local = int(np.argmin(band))
    divider_x_local = c0 + local

    darkest = float(band[local])
    median = float(np.median(band))
    if (median - darkest) < 8.0:
        return None

    return float(ix1 + divider_x_local)


def roboflow_fit_resize(img, size=MODEL_SIZE):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y, new_w, new_h


def map_mask_to_original(mask_reduced, pad_x, pad_y, new_w, new_h, orig_w, orig_h):
    binary_mask = (mask_reduced > 0.1).astype(np.uint8)
    cropped = binary_mask[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    if cropped.size == 0 or new_w <= 0 or new_h <= 0:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)
    restored = cv2.resize(
        cropped,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (restored * 255).astype(np.uint8)


def long_axis_from_rect(rect):
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90.0

    theta = math.radians(angle)
    ux, uy = math.cos(theta), math.sin(theta)
    
    p1 = np.array([cx - 0.5 * w * ux, cy - 0.5 * w * uy], dtype=np.float32)
    p2 = np.array([cx + 0.5 * w * ux, cy + 0.5 * w * uy], dtype=np.float32)
    
    return p1, p2, angle, w


def project_pixel_to_base(robot, px, py, bbox_long_side_px=None, frame_width=640):
    CAMERA_BASE_OFFSET_CM = 3.2 

    if bbox_long_side_px is not None and bbox_long_side_px > 0:
        CORRECTED_FOCAL = FOCAL_LENGTH * 1.25
        direct_dist_cm = ((CORRECTED_FOCAL * KNOWN_LENGTH) / bbox_long_side_px) + CAMERA_BASE_OFFSET_CM
    else:
        hit = robot.kine.project_to_ground(float(px), float(py)).copy()
        return float(hit[0][0]), float(hit[1][0])

    center_offset_px = px - (frame_width / 2.0)
    angle_rad = math.atan2(center_offset_px, FOCAL_LENGTH)

    x_mm = direct_dist_cm * 10.0 * math.cos(angle_rad)
    y_mm = direct_dist_cm * 10.0 * math.sin(angle_rad)

    return x_mm, y_mm


# ==========================================
# 3. UNIFIED STATE MACHINE PROGRAM
# ==========================================
class kickpickrefine(StateMachineProgram):

    def __init__(self):
        super().__init__()

        print("Initializing Fully Self-Contained Kick & Measure Pipeline...")
        
        # Standing Segmenter
        if os.path.exists(SEG_MODEL_PATH):
            self.segmenter = YOLO(SEG_MODEL_PATH)
        else:
            print(f"Warning: Segmenter weights '{SEG_MODEL_PATH}' not found!")
            self.segmenter = None

        # Standing Classifier
        self.classifier = self._load_classifier(CLS_MODEL_PATH)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Shared FSM Variables
        self.target_data = None
        self.turn_steps = 0
        self.last_charge_distance_mm = 0.0
        self.actual_retreated_mm = 0.0
        self.calculated_near_sideways_mm = 0.0
        self.calculated_near_turn_rad = 0.0

        # Calculated Motion Execution Parameters
        self.measured_forward_cm = 0.0
        self.measured_lateral_cm = 0.0
        self.measured_dividing_angle_deg = 0.0

        # Post-Turn Rescan Variables
        self.post_turn_sideways_mm = 0.0
        self.post_turn_angle_rad = 0.0

        # Fallen Model Setup
        self.fallen_model_path = None
        if os.path.exists(FALLEN_MODEL_PRIMARY):
            self.fallen_model_path = FALLEN_MODEL_PRIMARY
        elif os.path.exists(FALLEN_MODEL_FALLBACK):
            self.fallen_model_path = FALLEN_MODEL_FALLBACK

        self.fallen_model = None
        if self.fallen_model_path is not None:
            print(f"Loading fallen segmentation weights: {self.fallen_model_path}")
            self.fallen_model = YOLO(self.fallen_model_path)
            self.fallen_model.eval()
        else:
            print("Warning: No fallen segmentation weights found.")

        self.target = None
        self.debug_view = None

    def _load_classifier(self, path):
        try:
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            print(f"Classifier loaded successfully on {DEVICE}")
            return model
        except Exception as exc:
            print(f"ERROR: Failed to load classifier from '{path}': {exc}")
            return None

    def rotate_crop(self, image, obb):
        points = obb.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        center, size, angle = rect
        width, height = size

        if width <= 0 or height <= 0:
            return None, None, 0.0

        major_length = max(width, height)

        if width < height:
            angle += 90
            width, height = height, width

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )

        x, y = int(center[0]), int(center[1])
        pad = 20

        x1 = max(0, x - int(width / 2) - pad)
        y1 = max(0, y - int(height / 2) - pad)
        x2 = min(image.shape[1], x + int(width / 2) + pad)
        y2 = min(image.shape[0], y + int(height / 2) + pad)

        crop = rotated[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, None, 0.0

        return crop, (x1, y1, x2, y2), major_length

    def split_halves(self, image):
        try:
            if image is None or image.size == 0:
                return None, None
            h, w = image.shape[:2]
            mid = w // 2
            if mid < MIN_HALF_SIZE or (w - mid) < MIN_HALF_SIZE:
                return None, None
            return image[:, :mid].copy(), image[:, mid:].copy()
        except Exception as exc:
            print(f"split_halves failed: {exc}")
            return None, None

    def remove_white_border(self, image, threshold=WHITE_THRESHOLD, margin=BORDER_MARGIN):
        try:
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
        except Exception as exc:
            print(f"remove_white_border failed: {exc}")
            return image

    def preprocess_half(self, half_bgr):
        try:
            if half_bgr is None or half_bgr.size == 0 or half_bgr.shape[0] < MIN_HALF_SIZE or half_bgr.shape[1] < MIN_HALF_SIZE:
                return None
            gray = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 5, 75, 75)

            if self.clahe is None:
                self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = self.clahe.apply(filtered)

            three_channel = np.repeat(equalized[:, :, np.newaxis], 3, axis=2)
            pil_image = Image.fromarray(three_channel)
            tensor = PYTORCH_TENSOR_TRANSFORMS(pil_image)
            return tensor.unsqueeze(0)
        except Exception as exc:
            print(f"preprocess_half failed: {exc}")
            return None

    def predict_half(self, tensor):
        try:
            if self.classifier is None or tensor is None:
                return None, 0.0
            with torch.no_grad():
                tensor = tensor.to(DEVICE)
                logits = self.classifier(tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
                pred_class = int(predicted.item())
                conf_value = float(confidence.item())
            if pred_class < 0 or pred_class >= NUM_CLASSES:
                return None, 0.0
            return pred_class, conf_value
        except Exception as exc:
            print(f"predict_half failed: {exc}")
            return None, 0.0

    def measure_lateral_offset_mm(self, frame, conf=0.15):
        if frame is None or self.segmenter is None:
            return 0.0, False

        image_h, image_w = frame.shape[:2]
        results = self.segmenter(frame, conf=conf, verbose=False)
        if not results or len(results[0]) == 0:
            return 0.0, False

        result = results[0]
        box = None
        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes.xyxy.cpu().numpy()[0]

        if box is None:
            return 0.0, False

        x1_box, y1_box, x2_box, y2_box = box
        cx = float((x1_box + x2_box) * 0.5)
        divider_x = estimate_divider_x(frame, x1_box, y1_box, x2_box, y2_box)
        aim_x = float(divider_x) if divider_x is not None else cx

        aim_error_px = float(aim_x - (image_w * 0.5))
        focal_px = (image_w * 0.5) / math.tan(math.radians(CAMERA_HFOV_DEG * 0.5))
        lateral_angle_rad = math.atan2(aim_error_px, focal_px)

        error_sideways_mm = 50.0 * math.tan(lateral_angle_rad)
        error_sideways_mm -= 10.0

        return error_sideways_mm, True

    # ==========================================
    # 4. INTERNAL FSM NODES
    # ==========================================
    class ResetRun(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.turn_steps = 0
            self.parent.target_data = None
            self.parent.last_charge_distance_mm = 0.0
            self.parent.actual_retreated_mm = 0.0
            self.parent.calculated_near_sideways_mm = 0.0
            self.parent.calculated_near_turn_rad = 0.0
            self.parent.measured_forward_cm = 0.0
            self.parent.measured_lateral_cm = 0.0
            self.parent.measured_dividing_angle_deg = 0.0
            self.parent.post_turn_sideways_mm = 0.0
            self.parent.post_turn_angle_rad = 0.0
            self.post_completion()

    class TrackAndIdentify(StateNode):
        def start(self, event=None):
            super().start(event)
            frame = self.robot.camera_image

            if frame is None or self.parent.segmenter is None:
                self.parent.target_data = None
                self.post_failure()
                return

            image_h, image_w = frame.shape[:2]
            results = self.parent.segmenter(frame, conf=0.25, iou=0.5, agnostic_nms=True, verbose=False)

            if not results:
                self.parent.target_data = None
                self.post_failure()
                return

            result = results[0]
            candidates = []
            obb = getattr(result, "obb", None)
            boxes = getattr(result, "boxes", None)
            quads_info = []
            detected_indexes = set()

            if obb is not None and len(obb) > 0:
                raw_quads = list(obb.xyxyxyxy.cpu().numpy())
                for idx, q in enumerate(raw_quads):
                    quads_info.append((q, None))
                    detected_indexes.add(idx)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                for idx, box in enumerate(xyxy):
                    if idx not in detected_indexes:
                        x1b, y1b, x2b, y2b = box
                        quad = np.array([[x1b, y1b], [x2b, y1b], [x2b, y2b], [x1b, y2b]], dtype=np.float32)
                        pixel_length = max(x2b - x1b, y2b - y1b)
                        quads_info.append((quad, pixel_length))

            for raw_obb, fallback_length in quads_info:
                crop, bbox, pixel_length = self.parent.rotate_crop(frame, raw_obb)
                if pixel_length <= 0 and fallback_length is not None:
                    pixel_length = fallback_length

                if crop is None or crop.size == 0 or pixel_length <= 0:
                    continue

                h, w, _ = crop.shape
                if h < 50 or w < 50:
                    continue

                left_raw, right_raw = self.parent.split_halves(crop)
                if left_raw is None or right_raw is None:
                    continue

                left_clean = self.parent.remove_white_border(left_raw)
                right_clean = self.parent.remove_white_border(right_raw)

                if left_clean is None or right_clean is None or left_clean.size == 0 or right_clean.size == 0:
                    continue

                left_tensor = self.parent.preprocess_half(left_clean)
                right_tensor = self.parent.preprocess_half(right_clean)

                if left_tensor is None or right_tensor is None:
                    continue

                left_pred, left_conf = self.parent.predict_half(left_tensor)
                right_pred, right_conf = self.parent.predict_half(right_tensor)

                if left_pred is None or right_pred is None:
                    continue

                x1_box, y1_box, x2_box, y2_box = bbox
                cx = float((x1_box + x2_box) * 0.5)

                divider_x = estimate_divider_x(frame, x1_box, y1_box, x2_box, y2_box)
                aim_x = float(divider_x) if divider_x is not None else cx

                angle_deg, distance_cm = estimate_angle_and_distance_cm(aim_x, image_w, pixel_length)
                aim_error_px = float(aim_x - (image_w * 0.5))

                candidates.append({
                    "bbox": bbox,
                    "label": f"{left_pred}-{right_pred}",
                    "confidence": min(left_conf, right_conf),
                    "cx": cx,
                    "aim_x": aim_x,
                    "cy": float((y1_box + y2_box) * 0.5),
                    "angle_deg": angle_deg,
                    "distance_cm": distance_cm,
                    "aim_error_px": aim_error_px,
                    "using_black_line": (divider_x is not None)
                })

            if not candidates:
                self.parent.target_data = None
                self.post_failure()
                return

            view = frame.copy()
            valid_targets = []

            for cand in candidates:
                bx1, by1, bx2, by2 = map(int, cand["bbox"])
                if cand["label"] in ["3-4", "4-3"]:
                    box_color = (0, 255, 0)
                    label_text = f"TARGET: {cand['label']} ({cand['confidence']:.2f})"
                    valid_targets.append(cand)
                else:
                    box_color = (0, 0, 255)
                    label_text = f"IGNORE: {cand['label']}"

                cv2.rectangle(view, (bx1, by1), (bx2, by2), box_color, 2)
                aim_px = int(cand["aim_x"])
                cv2.line(view, (aim_px, by1), (aim_px, by2), (255, 0, 255), 2)
                cv2.putText(view, f"{label_text} - {cand['distance_cm']:.1f}cm", (bx1, max(15, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            imshow("Multi-Target Tracking Feed", view)

            if not valid_targets:
                self.parent.target_data = None
                self.post_failure()
                return

            best_target = max(valid_targets, key=lambda c: c["cy"])
            self.parent.target_data = best_target
            self.post_data(best_target)

    class CalculateTrajectory(StateNode):
        def start(self, event=None):
            super().start(event)
            target = self.parent.target_data
            if target is None:
                self.post_failure()
                return

            angle_deg = float(target["angle_deg"])
            aim_error_px = float(target["aim_error_px"])
            distance_cm = float(target["distance_cm"])

            align_tol_px = ALIGN_TOL_PX_NEAR if distance_cm <= NEAR_ALIGN_DISTANCE_CM else ALIGN_TOL_PX_FAR

            if abs(aim_error_px) > align_tol_px and self.parent.turn_steps < MAX_TURN_STEPS:
                turn_deg = clamp(-0.8 * angle_deg, -MAX_TURN_DEG, MAX_TURN_DEG)
                if abs(turn_deg) < MIN_TURN_DEG:
                    turn_deg = MIN_TURN_DEG * sign(-angle_deg)
                self.parent.turn_steps += 1
                self.post_data({"cmd": "turn", "angle_deg": float(turn_deg)})
                return

            print("[Trajectory] Heading aligned straight on domino center line.")
            self.parent.turn_steps = 0
            self.post_success()

    class ActuateMotion(ActionNode):
        def start(self, event=None):
            super().start(event)
            if not isinstance(event, DataEvent) or not isinstance(event.data, dict):
                self.post_failure()
                return

            cmd = event.data.get("cmd", "")
            if cmd == "turn":
                deg = float(event.data.get("angle_deg", 0.0))
                self.robot.actuators["drive"].turn(self, math.radians(deg), None)
            else:
                self.post_failure()

    class ApproachCheckpoint(ActionNode):
        def start(self, event=None):
            super().start(event)
            target = self.parent.target_data
            if target is None:
                self.post_failure()
                return

            total_distance_mm = float(target["distance_cm"]) * 10.0
            self.parent.last_charge_distance_mm = total_distance_mm

            approach_distance_mm = max(0.0, total_distance_mm - 70.0)
            print(f"[Approach Checkpoint] Driving {approach_distance_mm:.1f}mm to close zone...")
            self.robot.actuators["drive"].forward(self, approach_distance_mm)

    class ScanNearZoneDrift(StateNode):
        def start(self, event=None):
            super().start(event)
            time.sleep(0.15)

            self.parent.calculated_near_sideways_mm = 0.0
            self.parent.calculated_near_turn_rad = 0.0

            frame = self.robot.camera_image
            if frame is None:
                print("[Scan Near Zone] Frame empty. Skipping close adjustments.")
                self.post_success()
                return

            image_h, image_w = frame.shape[:2]
            base_line_slope_rad = 0.0
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)

                mask = np.zeros_like(edges)
                mask[int(image_h * 0.45):int(image_h * 0.95), :] = 255
                masked_edges = cv2.bitwise_and(edges, mask)

                pts = np.argwhere(masked_edges > 0)
                if len(pts) > 20:
                    pts = np.fliplr(pts)
                    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                    base_line_slope_rad = math.atan2(vy[0], vx[0])
            except Exception as pixel_err:
                base_line_slope_rad = 0.0

            error_sideways_mm, _ = self.parent.measure_lateral_offset_mm(frame, conf=0.15)

            self.parent.calculated_near_turn_rad = base_line_slope_rad
            self.parent.calculated_near_sideways_mm = error_sideways_mm
            self.post_data({"cmd": "execute_readjustment"})

    class TurnParallelNear(ActionNode):
        def start(self, event=None):
            super().start(event)
            base_slope_rad = self.parent.calculated_near_turn_rad
            base_slope_deg = math.degrees(base_slope_rad)

            if abs(base_slope_deg) > 0.4:
                print(f"[Readjust Step 1] Turning chassis {-base_slope_deg:.2f}°...")
                self.robot.actuators["drive"].turn(self, -base_slope_rad, None)
            else:
                self.post_completion()

    class ShiftLeft5cm(ActionNode):
        """Angle readjustment ke baad 5 cm (50 mm) Left shift karta hai."""
        def start(self, event=None):
            super().start(event)
            print("[Strike Sequence 1/4] Shifting 5 cm (50 mm) LEFT...")
            self.robot.actuators["drive"].sideways(self, 50.0, None)

    class ForwardKnock60mm(ActionNode):
        """60 mm forward aage badhke domino ko strike/knock karta hai."""
        def start(self, event=None):
            super().start(event)
            print("[Strike Sequence 2/4] Driving 60.0 mm FORWARD to knock domino...")
            self.robot.actuators["drive"].forward(self, 60.0, None)

    class Retreat60mm(ActionNode):
        """60 mm peeche aata hai."""
        def start(self, event=None):
            super().start(event)
            print("[Strike Sequence 3/4] Driving 60.0 mm BACKWARD...")
            self.parent.actual_retreated_mm = 60.0
            self.robot.actuators["drive"].forward(self, -60.0, None)

    class ShiftRight50cm(ActionNode):
        """50 cm (500 mm) Right shift karta hai."""
        def start(self, event=None):
            super().start(event)
            print("[Strike Sequence 4/4] Shifting 50 cm (500 mm) RIGHT...")
            self.robot.actuators["drive"].sideways(self, -50.0, None)

    class PreparePickupRun(StateNode):
        def start(self, event=None):
            super().start(event)
            print("\n==============================================")
            print(">>> [SCANNING FALLEN DOMINO POSE & DISTANCE] <<<")
            print("==============================================")
            self.parent.target = None
            self.parent.debug_view = None
            self.post_completion()

    class DetectTarget(StateNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.fallen_model is None:
                print("[FALLEN-DETECT] ERROR: Fallen model is missing/None!")
                self.post_failure()
                return

            time.sleep(SETTLE_DELAY_SEC)

            image = None
            results = []
            for attempt in range(DETECT_TRIES):
                image = self.robot.camera_image
                if image is None:
                    time.sleep(DETECT_RETRY_DELAY_SEC)
                    continue

                canvas, scale, pad_x, pad_y, new_w, new_h = roboflow_fit_resize(image, MODEL_SIZE)
                try:
                    results = self.parent.fallen_model.predict(
                        source=canvas,
                        conf=PREDICT_CONF,
                        iou=PREDICT_IOU,
                        verbose=False,
                    )
                except Exception as ex:
                    print(f"[FALLEN-DETECT] Predict failed: {ex}")
                    results = []

                if len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                    break
                time.sleep(DETECT_RETRY_DELAY_SEC)

            if image is None or len(results) == 0 or results[0].masks is None:
                print("[FALLEN-DETECT] FAILED: No fallen domino detected.")
                self.post_failure()
                return

            h, w = image.shape[:2]
            view = image.copy()

            result = results[0]
            mask_data = result.masks.data
            candidates = []

            for i in range(len(mask_data)):
                mask_small = mask_data[i].detach().cpu().numpy()
                if np.count_nonzero(mask_small > 0.1) == 0:
                    continue

                mask_orig = map_mask_to_original(mask_small, pad_x, pad_y, new_w, new_h, w, h)
                mask_area = float(cv2.countNonZero(mask_orig))
                contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    small_contours, _ = cv2.findContours((mask_small > 0.1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not small_contours:
                        continue
                    cnt_small = max(small_contours, key=cv2.contourArea)
                    rect_small = cv2.minAreaRect(cnt_small)
                    (cx_s, cy_s), (rw_s, rh_s), ang_s = rect_small
                    cx = (cx_s - pad_x) / scale
                    cy = (cy_s - pad_y) / scale
                    rect = ((cx, cy), (rw_s / scale, rh_s / scale), ang_s)
                    mask_area = float(cv2.contourArea(cnt_small)) / (scale * scale)
                else:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), _, _ = rect

                candidates.append({
                    "mask_area": mask_area,
                    "cx": cx,
                    "cy": cy,
                    "rect": rect
                })

            if len(candidates) == 0:
                print("[FALLEN-DETECT] FAILED: Zero mask candidates.")
                self.post_failure()
                return

            best = max(candidates, key=lambda c: c["mask_area"])
            _, _, orientation_deg, long_side_px = long_axis_from_rect(best["rect"])

            try:
                center_x_mm, center_y_mm = project_pixel_to_base(
                    self.robot, 
                    best["cx"], 
                    best["cy"], 
                    bbox_long_side_px=long_side_px, 
                    frame_width=w
                )
            except Exception as ex:
                print(f"[FALLEN-DETECT] Projection Error: {ex}")
                self.post_failure()
                return

            retreat_cm = self.parent.actual_retreated_mm / 10.0
            camera_forward_cm = center_x_mm / 10.0
            
            self.parent.measured_lateral_cm = center_y_mm / 10.0
            self.parent.measured_forward_cm = retreat_cm + camera_forward_cm
            self.parent.measured_dividing_angle_deg = wrap_deg(orientation_deg)

            total_distance_cm = math.hypot(self.parent.measured_forward_cm, self.parent.measured_lateral_cm)

            print("\n" + "="*50)
            print("         FALLEN DOMINO DETECTION METRICS        ")
            print("="*50)
            print(f" ► Direct Distance : {total_distance_cm:.2f} cm")
            print(f" ► Forward Distance: {self.parent.measured_forward_cm:.2f} cm")
            print(f" ► Lateral Offset  : {self.parent.measured_lateral_cm:.2f} cm")
            print(f" ► Dividing Angle  : {self.parent.measured_dividing_angle_deg:.2f}°")
            print("="*50 + "\n")

            box = cv2.boxPoints(best["rect"]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(view, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(view, (int(best["cx"]), int(best["cy"])), 4, (255, 0, 0), -1)
            imshow("fallen_domino_measurement", view)

            self.post_success()

    # ==========================================
    # 5. STEPWISE POST-MEASUREMENT MOTION NODES
    # ==========================================
    class Step1_MoveLateralOffset(ActionNode):
        """Step 1: Move by the measured lateral offset (sideways)"""
        def start(self, event=None):
            super().start(event)
            lat_mm = self.parent.measured_lateral_cm * 10.0 + 50.0
            print(f"\n[STEP 1/5] Moving Lateral Offset: {lat_mm:.1f}mm")
            self.robot.actuators["drive"].sideways(self, lat_mm, None)

    class Step2_MoveForwardDistancePlus2(ActionNode):
        """Step 2: Move forward distance + 2 cm (in mm)"""
        def start(self, event=None):
            super().start(event)
            fwd_target_mm = (self.parent.measured_forward_cm + 2.0) * 10.0 + 20.0
            print(f"[STEP 2/5] Moving Forward (Measured + 2cm): {fwd_target_mm:.1f}mm")
            self.robot.actuators["drive"].forward(self, fwd_target_mm, None)

    class Step3_TurnNegativeComplementaryAngle(ActionNode):
        """Step 3: Turn the negative of the complementary angle of the dividing angle"""
        def start(self, event=None):
            super().start(event)
            div_angle_deg = self.parent.measured_dividing_angle_deg
            comp_angle_deg = 90.0 - div_angle_deg
            neg_comp_angle_deg = -comp_angle_deg
            
            neg_comp_rad = math.radians(neg_comp_angle_deg)
            print(f"[STEP 3/5] Turning Negative Complementary Angle: {neg_comp_angle_deg:.2f}° ({neg_comp_rad:.3f} rad)")
            self.robot.actuators["drive"].turn(self, neg_comp_rad, None)

    # ==========================================
    # NEW: POST-TURN RE-SCAN & FINE ALIGNMENT NODES
    # ==========================================
    class RescanFallenClose(StateNode):
        """Turn lene ke baad fallen domino ki black line ko firse scan karta hai"""
        def start(self, event=None):
            super().start(event)
            time.sleep(0.15)
            print("\n[POST-TURN RESCAN] Scanning fallen domino black line at close range...")

            frame = self.robot.camera_image
            if frame is None:
                print("[POST-TURN RESCAN] Camera frame empty. Proceeding to final drive...")
                self.parent.post_turn_sideways_mm = 0.0
                self.parent.post_turn_angle_rad = 0.0
                self.post_completion()
                return

            image_h, image_w = frame.shape[:2]
            line_angle_rad = 0.0
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)

                mask = np.zeros_like(edges)
                mask[int(image_h * 0.35):int(image_h * 0.95), :] = 255
                masked_edges = cv2.bitwise_and(edges, mask)

                pts = np.argwhere(masked_edges > 0)
                if len(pts) > 20:
                    pts = np.fliplr(pts)
                    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                    line_angle_rad = math.atan2(vy[0], vx[0])
            except Exception as err:
                line_angle_rad = 0.0

            error_sideways_mm, detected = self.parent.measure_lateral_offset_mm(frame, conf=0.15)
            if not detected:
                error_sideways_mm = 0.0

            self.parent.post_turn_sideways_mm = error_sideways_mm
            self.parent.post_turn_angle_rad = line_angle_rad
            print(f"[POST-TURN RESCAN] Measured Lateral Error: {error_sideways_mm:.1f}mm, Line Drift: {math.degrees(line_angle_rad):.2f}°")
            self.post_completion()

    class FineTuneSideways(ActionNode):
        """Pehle sideways shift hokar domino centre line align karta hai"""
        def start(self, event=None):
            super().start(event)
            sideways_mm = self.parent.post_turn_sideways_mm
            if abs(sideways_mm) > 1.0:
                print(f"[FINE-TUNE 1/2] Shifting sideways by {-sideways_mm:.1f}mm...")
                self.robot.actuators["drive"].sideways(self, -sideways_mm, None)
            else:
                self.post_completion()

    class FineTuneParallelAngle(ActionNode):
        """Fir angle adjust karke fallen domino ki black line ke parallel alignment lata hai"""
        def start(self, event=None):
            super().start(event)
            angle_rad = self.parent.post_turn_angle_rad
            angle_deg = math.degrees(angle_rad)
            if abs(angle_deg) > 0.4:
                print(f"[FINE-TUNE 2/2] Adjusting chassis angle by {-angle_deg:.2f}° to align parallel...")
                self.robot.actuators["drive"].turn(self, -angle_rad, None)
            else:
                self.post_completion()

    class Step4_Move10cmForward(ActionNode):
        """Step 4: Alignment complete hone ke baad last ke cm forward drive karta hai"""
        def start(self, event=None):
            super().start(event)
            print("[STEP 4/5] Alignment complete. Driving forward 150.0mm (15 cm)...")
            self.robot.actuators["drive"].forward(self, 150.0, None)

    class Step5_FinalKick(StateNode):
        """Step 5: Active Kick Action via PlaceKick().now()"""
        def start(self, event=None):
            super().start(event)
            print("[STEP 5/5] Executing PlaceKick().now()...")
            try:
                PlaceKick().now()
            except Exception as e:
                print(f"[STEP 5/5] PlaceKick execution warning: {e}")
            self.post_completion()

    # ==========================================
    # 6. FSM PIPELINE SETUP
    # ==========================================
    def setup(self):
        begin = StateNode().set_name("begin").set_parent(self)
        reset = self.ResetRun().set_name("reset").set_parent(self)
        track = self.TrackAndIdentify().set_name("track").set_parent(self)
        analyze = self.CalculateTrajectory().set_name("analyze").set_parent(self)
        move = self.ActuateMotion().set_name("move").set_parent(self)

        approach_chk = self.ApproachCheckpoint().set_name("approach_chk").set_parent(self)
        scan_near = self.ScanNearZoneDrift().set_name("scan_near").set_parent(self)
        turn_parallel = self.TurnParallelNear().set_name("turn_parallel").set_parent(self)

        # Knock & Retreat Sequence
        shift_left = self.ShiftLeft5cm().set_name("shift_left").set_parent(self)
        knock_fwd = self.ForwardKnock60mm().set_name("knock_fwd").set_parent(self)
        retreat_back = self.Retreat60mm().set_name("retreat_back").set_parent(self)
        shift_right = self.ShiftRight50cm().set_name("shift_right").set_parent(self)

        wait_after_kick = StateNode().set_name("wait_after_kick").set_parent(self)

        prep_pickup = self.PreparePickupRun().set_name("prep_pickup").set_parent(self)
        detect = self.DetectTarget().set_name("detect").set_parent(self)

        # Post-Measurement Motion Sequence Nodes
        m_step1 = self.Step1_MoveLateralOffset().set_name("m_step1").set_parent(self)
        m_step2 = self.Step2_MoveForwardDistancePlus2().set_name("m_step2").set_parent(self)
        m_step3 = self.Step3_TurnNegativeComplementaryAngle().set_name("m_step3").set_parent(self)

        # Post-Turn Rescan & Fine Alignment Nodes
        rescan_close = self.RescanFallenClose().set_name("rescan_close").set_parent(self)
        fine_sideways = self.FineTuneSideways().set_name("fine_sideways").set_parent(self)
        fine_parallel = self.FineTuneParallelAngle().set_name("fine_parallel").set_parent(self)

        m_step4 = self.Step4_Move10cmForward().set_name("m_step4").set_parent(self)
        m_step5 = self.Step5_FinalKick().set_name("m_step5").set_parent(self)

        done_print = Print("Sequence completed successfully!").set_name("done_print").set_parent(self)
        wait_scan = StateNode().set_name("wait_scan").set_parent(self)

        # Transitions
        TimerTrans(0.5).add_sources(begin).add_destinations(reset)
        CompletionTrans().add_sources(reset).add_destinations(track)

        DataTrans().add_sources(track).add_destinations(analyze)
        FailureTrans().add_sources(track).add_destinations(wait_scan)
        TimerTrans(1.0).add_sources(wait_scan).add_destinations(track)

        DataTrans().add_sources(analyze).add_destinations(move)
        SuccessTrans().add_sources(analyze).add_destinations(approach_chk)
        FailureTrans().add_sources(analyze).add_destinations(wait_scan)

        CompletionTrans().add_sources(move).add_destinations(track)
        FailureTrans().add_sources(move).add_destinations(wait_scan)

        CompletionTrans().add_sources(approach_chk).add_destinations(scan_near)
        FailureTrans().add_sources(approach_chk).add_destinations(wait_scan)

        # Near Zone Readjustment -> Shift Left -> Forward 60mm -> Back 60mm -> Shift Right 50cm
        DataTrans().add_sources(scan_near).add_destinations(turn_parallel)
        SuccessTrans().add_sources(scan_near).add_destinations(shift_left)
        FailureTrans().add_sources(scan_near).add_destinations(wait_scan)

        CompletionTrans().add_sources(turn_parallel).add_destinations(shift_left)
        FailureTrans().add_sources(turn_parallel).add_destinations(wait_scan)

        CompletionTrans().add_sources(shift_left).add_destinations(knock_fwd)
        FailureTrans().add_sources(shift_left).add_destinations(wait_scan)

        CompletionTrans().add_sources(knock_fwd).add_destinations(retreat_back)
        FailureTrans().add_sources(knock_fwd).add_destinations(wait_scan)

        CompletionTrans().add_sources(retreat_back).add_destinations(shift_right)
        FailureTrans().add_sources(retreat_back).add_destinations(wait_scan)

        CompletionTrans().add_sources(shift_right).add_destinations(wait_after_kick)
        FailureTrans().add_sources(shift_right).add_destinations(wait_scan)

        TimerTrans(1.5).add_sources(wait_after_kick).add_destinations(prep_pickup)

        # Measurement -> STEPWISE MOTION FLOW
        CompletionTrans().add_sources(prep_pickup).add_destinations(detect)
        
        # Detect -> Step 1 -> Step 2 -> Step 3 (Turn) -> Rescan -> Sideways Align -> Parallel Angle Align -> Step 4 -> Step 5
        SuccessTrans().add_sources(detect).add_destinations(m_step1)
        CompletionTrans().add_sources(m_step1).add_destinations(m_step2)
        CompletionTrans().add_sources(m_step2).add_destinations(m_step3)
        
        # Turn hone ke baad re-scanning & fine alignment
        CompletionTrans().add_sources(m_step3).add_destinations(rescan_close)
        CompletionTrans().add_sources(rescan_close).add_destinations(fine_sideways)
        CompletionTrans().add_sources(fine_sideways).add_destinations(fine_parallel)
        CompletionTrans().add_sources(fine_parallel).add_destinations(m_step4)

        CompletionTrans().add_sources(m_step4).add_destinations(m_step5)
        CompletionTrans().add_sources(m_step5).add_destinations(done_print)
        FailureTrans().add_sources(detect).add_destinations(done_print)

        NullTrans().add_sources(done_print).add_destinations(reset)

        return self
import os
import cv2
import time
import numpy as np
from aim_fsm import *
from ultralytics import YOLO

# ==========================================
# 1. LOAD VISION MODELS
# ==========================================
try:
    fallen_model = YOLO("fallen.pt")
    different_model = YOLO("different.pt")
    print("[VISION] Successfully loaded 'fallen.pt' and 'different.pt' models.")
except Exception as e:
    print(f"[VISION ERROR] Model load failed: {e}")
    fallen_model = None
    different_model = None


# ==========================================
# 2. BOARD STATE
# ==========================================
class BoardState:
    def __init__(self):
        self.open_ends = {}

    def set_open_end(self, end_id, domino_val, dist_mm):
        self.open_ends[end_id] = {"value": domino_val, "dist_mm": dist_mm}
        print(f"\n[BOARD GRAPH] Open End '{end_id}' -> Classified: [{domino_val}] at {dist_mm}mm")

board = BoardState()


# ==========================================
# 3. HELPER: DUAL-HALF CLASSIFICATION
# ==========================================
def classify_fallen_domino(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return "Unknown"

    h, w, _ = crop.shape

    # Detect orienting black line across major axis
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    if h > w:
        half1 = crop[0:h//2, :]
        half2 = crop[h//2:h, :]
    else:
        half1 = crop[:, 0:w//2]
        half2 = crop[:, w//2:w]

    val1, val2 = "?", "?"
    if different_model is not None:
        res1 = different_model(half1, verbose=False)
        res2 = different_model(half2, verbose=False)
        
        if len(res1[0].boxes) > 0:
            val1 = res1[0].names[int(res1[0].boxes.cls[0])]
        if len(res2[0].boxes) > 0:
            val2 = res2[0].names[int(res2[0].boxes.cls[0])]

    return f"{val1}-{val2}"


# ==========================================
# 4. FSM NODES
# ==========================================

class WaitForHumanSignal(StateNode):
    def start(self, event=None):
        super().start(event)
        print("\n=======================================================")
        print(">>> Waiting for human to place domino.                 <<<")
        print(">>> Type 'done' in terminal once domino is placed.     <<<")
        print("=======================================================")
        
        user_input = input("User input ('done'): ").strip().lower()
        while user_input != "done":
            user_input = input("Please type 'done': ").strip().lower()
        
        print("[SIGNAL] 'done' received! Locating domino...")
        self.post_success()


class DriveToStandoff(Forward):
    """
    Subclasses Forward action node directly to safely handle 
    actuator lock/unlock mechanics automatically.
    """
    def __init__(self, distance_mm=120.0):
        super().__init__(distance_mm)

    def start(self, event=None):
        print(f"\n[NAV] Driving forward to reach 3 cm close-up standoff...")
        super().start(event)


class ScanAndClassifyDomino(StateNode):
    """
    Executes after the drive completes and actuator lock is released.
    Scans the fallen domino at close range (3cm) and classifies both halves.
    """
    def start(self, event=None):
        super().start(event)
        print("[SCAN] Robot in position! Capturing close-up frame...")
        
        close_frame = self.robot.camera_image
        if close_frame is not None and fallen_model is not None:
            results = fallen_model(close_frame, verbose=False)
            if len(results[0].boxes) > 0:
                target_box = results[0].boxes.xyxy[0].cpu().numpy()
                classified_value = classify_fallen_domino(close_frame, target_box)
                board.set_open_end("END_A", domino_val=classified_value, dist_mm=30.0)
            else:
                print("[SCAN WARNING] No fallen domino detected at 3cm standoff!")
        else:
            print("[SCAN WARNING] Camera frame or models uninitialized!")

        self.post_success()


# ==========================================
# 5. MASTER FSM
# ==========================================
class domino_game_master(StateMachineProgram):
    def setup(self):
        begin = StateNode().set_name("begin").set_parent(self)
        
        wait_human = WaitForHumanSignal().set_name("wait_human").set_parent(self)
        
        # Built-in Forward action handles drive actuator locking cleanly
        drive_3cm = DriveToStandoff(120.0).set_name("drive_3cm").set_parent(self)
        
        scan_domino = ScanAndClassifyDomino().set_name("scan_domino").set_parent(self)
        
        done_step = Print("Scan complete. Waiting for next turn.\n").set_name("done_step").set_parent(self)

        # Transitions
        TimerTrans(0.2).add_sources(begin).add_destinations(wait_human)
        
        # 1. Human types 'done' -> Drive to 3cm
        SuccessTrans().add_sources(wait_human).add_destinations(drive_3cm)
        
        # 2. Drive Action Finishes (CompletionTrans unlocks actuator) -> Scan & Classify
        CompletionTrans().add_sources(drive_3cm).add_destinations(scan_domino)
        
        # 3. Scan completes -> Print & Loop back to human turn
        SuccessTrans().add_sources(scan_domino).add_destinations(done_step)
        FailureTrans().add_sources(scan_domino).add_destinations(done_step)
        
        NullTrans().add_sources(done_step).add_destinations(wait_human)

        return self
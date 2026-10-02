#!/usr/bin/env python3
"""Small OpenCV GUI for four-corner domino-face annotation.

The tool is intentionally tailored to the ten single-domino images copied into
the standalone annotation folder.  It autosaves JSON after every accepted tile
and exports a perspective-rectified crop for visual verification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WINDOW_NAME = "Domino Oracle-Crop Annotator"
CANVAS_WIDTH = 1440
CANVAS_HEIGHT = 860
IMAGE_PANEL = (20, 70, 920, 820)
PREVIEW_SIZE = (430, 215)
RECTIFIED_SIZE = (600, 300)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def order_quad(points: list[list[float]] | np.ndarray) -> np.ndarray:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if len(np.unique(np.round(pts, 3), axis=0)) != 4:
        raise ValueError("The four corners must be different points.")
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[int(np.argmin(sums))]
    ordered[2] = pts[int(np.argmax(sums))]
    ordered[1] = pts[int(np.argmin(diffs))]
    ordered[3] = pts[int(np.argmax(diffs))]
    if len(np.unique(np.round(ordered, 3), axis=0)) != 4:
        # Fallback for unusually rotated perspective quads.
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        cycle = pts[np.argsort(angles)]
        start = int(np.argmin(cycle.sum(axis=1)))
        ordered = np.roll(cycle, -start, axis=0)
        if cv2.contourArea(ordered.reshape(-1, 1, 2), oriented=True) < 0:
            ordered = ordered[[0, 3, 2, 1]]
    area = abs(float(cv2.contourArea(ordered.reshape(-1, 1, 2))))
    if area < 100.0:
        raise ValueError("The selected quadrilateral is too small or collapsed.")
    return ordered


def rotate_quad_180(ordered: np.ndarray) -> np.ndarray:
    """Keep TL/TR/BR/BL destination order while rotating its source by 180°."""
    return np.asarray(ordered, dtype=np.float32)[[2, 3, 0, 1]]


def rectify(image_bgr: np.ndarray, ordered_quad: np.ndarray) -> np.ndarray:
    width, height = RECTIFIED_SIZE
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.asarray(ordered_quad, dtype=np.float32), destination
    )
    return cv2.warpPerspective(
        image_bgr,
        transform,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class DominoAnnotator:
    def __init__(self, images_dir: Path, output_dir: Path) -> None:
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.annotation_path = output_dir / "annotations.json"
        self.crops_dir = output_dir / "crops"
        self.images = sorted(
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"No JPG or PNG images found in {images_dir}")
        if self.annotation_path.exists():
            self.data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        else:
            self.data = {"version": 1, "images": {}}
        self.index = 0
        self.image_bgr: np.ndarray | None = None
        self.points: list[list[float]] = []
        self.ordered_quad: np.ndarray | None = None
        self.rotation_180 = False
        self.left_pips: int | None = None
        self.right_pips: int | None = None
        self.pip_entry_side = 0
        self.posture = "standing"
        self.message = "Click the four visible face corners. Order does not matter."
        self.image_scale = 1.0
        self.image_offset = (0, 0)
        self.display_size = (1, 1)
        self.finished = False
        self._load_current()

    @property
    def image_path(self) -> Path:
        return self.images[self.index]

    def _load_current(self) -> None:
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read {self.image_path}")
        self.image_bgr = image
        self._reset_current()
        saved = self._saved_domino()
        if saved:
            self.message = "Already annotated. Press D to redo, ] for next, Q to quit."

    def _reset_current(self) -> None:
        self.points = []
        self.ordered_quad = None
        self.rotation_180 = False
        self.left_pips = None
        self.right_pips = None
        self.pip_entry_side = 0
        self.posture = "standing"
        self.message = "Click the four visible face corners. Order does not matter."

    def _saved_domino(self) -> dict[str, Any] | None:
        entry = self.data.get("images", {}).get(self.image_path.name, {})
        dominoes = entry.get("dominoes", [])
        return dominoes[0] if dominoes else None

    def _fit_image(self) -> np.ndarray:
        assert self.image_bgr is not None
        x0, y0, x1, y1 = IMAGE_PANEL
        available_w = x1 - x0
        available_h = y1 - y0
        height, width = self.image_bgr.shape[:2]
        self.image_scale = min(available_w / width, available_h / height)
        display_w = max(1, int(round(width * self.image_scale)))
        display_h = max(1, int(round(height * self.image_scale)))
        resized = cv2.resize(
            self.image_bgr,
            (display_w, display_h),
            interpolation=cv2.INTER_AREA,
        )
        offset_x = x0 + (available_w - display_w) // 2
        offset_y = y0 + (available_h - display_h) // 2
        self.image_offset = (offset_x, offset_y)
        self.display_size = (display_w, display_h)
        return resized

    def _image_to_canvas(self, point: list[float] | tuple[float, float]) -> tuple[int, int]:
        return (
            int(round(self.image_offset[0] + point[0] * self.image_scale)),
            int(round(self.image_offset[1] + point[1] * self.image_scale)),
        )

    def _canvas_to_image(self, x: int, y: int) -> list[float] | None:
        ox, oy = self.image_offset
        display_w, display_h = self.display_size
        if not (ox <= x < ox + display_w and oy <= y < oy + display_h):
            return None
        assert self.image_bgr is not None
        image_x = (x - ox) / self.image_scale
        image_y = (y - oy) / self.image_scale
        height, width = self.image_bgr.shape[:2]
        return [
            float(np.clip(image_x, 0, width - 1)),
            float(np.clip(image_y, 0, height - 1)),
        ]

    def _draw_polygon(
        self,
        canvas: np.ndarray,
        points: list[list[float]] | np.ndarray,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        display_points = np.array(
            [self._image_to_canvas(point) for point in points], dtype=np.int32
        )
        if len(display_points) >= 2:
            cv2.polylines(
                canvas,
                [display_points.reshape(-1, 1, 2)],
                len(display_points) == 4,
                color,
                3,
                cv2.LINE_AA,
            )
        for number, point in enumerate(display_points):
            cv2.circle(canvas, tuple(point), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(number),
                (int(point[0] + 8), int(point[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        if len(display_points):
            origin = tuple(display_points[0] + np.array([10, 24]))
            cv2.putText(
                canvas,
                label,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    def _preview(self) -> np.ndarray | None:
        if self.ordered_quad is None or self.image_bgr is None:
            return None
        quad = (
            rotate_quad_180(self.ordered_quad)
            if self.rotation_180
            else self.ordered_quad
        )
        preview = rectify(self.image_bgr, quad)
        cv2.line(
            preview,
            (preview.shape[1] // 2, 0),
            (preview.shape[1] // 2, preview.shape[0] - 1),
            (0, 220, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(preview, "LEFT", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(
            preview,
            "RIGHT",
            (preview.shape[1] - 115, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        return preview

    def render(self) -> np.ndarray:
        canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 32, dtype=np.uint8)
        cv2.putText(
            canvas,
            f"Image {self.index + 1}/{len(self.images)}: {self.image_path.name}",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        resized = self._fit_image()
        ox, oy = self.image_offset
        display_w, display_h = self.display_size
        canvas[oy : oy + display_h, ox : ox + display_w] = resized

        saved = self._saved_domino()
        if saved:
            pips = saved.get("pips", ["?", "?"])
            self._draw_polygon(
                canvas,
                saved["quad"],
                (60, 220, 80),
                f"SAVED {pips[0]}-{pips[1]} {saved.get('posture', '?')}",
            )
        if self.points:
            self._draw_polygon(canvas, self.points, (0, 220, 255), "CURRENT")

        panel_x = 955
        cv2.rectangle(canvas, (940, 55), (1425, 840), (70, 70, 70), 2)
        cv2.putText(canvas, "RECTIFIED PREVIEW", (panel_x, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
        preview = self._preview()
        if preview is not None:
            preview_resized = cv2.resize(preview, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
            canvas[105 : 105 + PREVIEW_SIZE[1], panel_x : panel_x + PREVIEW_SIZE[0]] = preview_resized
        else:
            cv2.rectangle(
                canvas,
                (panel_x, 105),
                (panel_x + PREVIEW_SIZE[0], 105 + PREVIEW_SIZE[1]),
                (85, 85, 85),
                -1,
            )
            cv2.putText(canvas, "Appears after 4 clicks", (panel_x + 65, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)

        left = "?" if self.left_pips is None else str(self.left_pips)
        right = "?" if self.right_pips is None else str(self.right_pips)
        info_lines = [
            f"Points: {len(self.points)}/4",
            f"Pips in preview: {left} - {right}",
            f"Posture: {self.posture}",
            "",
            "After 4 clicks:",
            "  type LEFT pip 0-6",
            "  then RIGHT pip 0-6",
            "",
            "S / F    standing / fallen",
            "R        rotate preview 180 deg",
            "Backspace undo last corner",
            "C        clear current selection",
            "Enter    save tile and next image",
            "D        redo saved annotation",
            "[ / ]    previous / next image",
            "Q        save and quit",
        ]
        y = 360
        for line in info_lines:
            cv2.putText(canvas, line, (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 1, cv2.LINE_AA)
            y += 29
        annotated = sum(
            bool(entry.get("dominoes"))
            for entry in self.data.get("images", {}).values()
        )
        cv2.putText(
            canvas,
            f"Progress: {annotated}/{len(self.images)} images saved",
            (panel_x, 805),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 230, 110),
            2,
            cv2.LINE_AA,
        )
        message_color = (70, 230, 255) if not self.finished else (80, 255, 100)
        cv2.putText(canvas, self.message[:105], (24, 848), cv2.FONT_HERSHEY_SIMPLEX, 0.58, message_color, 2, cv2.LINE_AA)
        return canvas

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or self._saved_domino():
            return
        if len(self.points) >= 4:
            self.message = "Four corners selected. Type the left and right pip digits."
            return
        point = self._canvas_to_image(x, y)
        if point is None:
            self.message = "Click inside the photo, directly on a domino-face corner."
            return
        self.points.append(point)
        if len(self.points) == 4:
            try:
                self.ordered_quad = order_quad(self.points)
                self.points = self.ordered_quad.astype(float).tolist()
                self.message = "Preview ready. Type LEFT pip count (0-6)."
            except ValueError as exc:
                self.message = str(exc) + " Press C and try again."
                self.ordered_quad = None
        else:
            self.message = f"Corner {len(self.points)}/4 saved. Click the next corner."

    def _save_annotation(self) -> None:
        if self.ordered_quad is None or len(self.points) != 4:
            self.message = "Select four corners before saving."
            return
        if self.left_pips is None or self.right_pips is None:
            self.message = "Enter both pip counts with number keys 0-6."
            return
        assert self.image_bgr is not None
        final_quad = (
            rotate_quad_180(self.ordered_quad)
            if self.rotation_180
            else self.ordered_quad
        )
        annotation = {
            "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in final_quad],
            "pips": [int(self.left_pips), int(self.right_pips)],
            "posture": self.posture,
        }
        height, width = self.image_bgr.shape[:2]
        self.data.setdefault("images", {})[self.image_path.name] = {
            "width": width,
            "height": height,
            "dominoes": [annotation],
        }
        _atomic_write_json(self.annotation_path, self.data)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = self.crops_dir / f"{self.image_path.stem}-domino-0.png"
        cv2.imwrite(str(crop_path), rectify(self.image_bgr, final_quad))
        if self.index + 1 < len(self.images):
            self.index += 1
            self._load_current()
        else:
            self.finished = True
            self.message = "ALL IMAGES ANNOTATED. Press Q to close."

    def _delete_saved_annotation(self) -> None:
        images = self.data.setdefault("images", {})
        if self.image_path.name in images:
            images[self.image_path.name] = {
                "width": int(self.image_bgr.shape[1]),
                "height": int(self.image_bgr.shape[0]),
                "dominoes": [],
            }
            _atomic_write_json(self.annotation_path, self.data)
        self._reset_current()
        self.message = "Saved annotation cleared in JSON. Click four corners to redo it."

    def handle_key(self, key: int) -> bool:
        low = key & 0xFF
        if low in (ord("q"), ord("Q"), 27):
            _atomic_write_json(self.annotation_path, self.data)
            return False
        if low in (8, 127):
            if self.points and self.ordered_quad is None:
                self.points.pop()
                self.message = f"Undid corner. {len(self.points)}/4 remain."
            else:
                self.message = "Press C to clear a completed selection."
        elif low in (ord("c"), ord("C")):
            if not self._saved_domino():
                self._reset_current()
        elif low in (ord("d"), ord("D")):
            self._delete_saved_annotation()
        elif low in (ord("s"), ord("S")):
            self.posture = "standing"
        elif low in (ord("f"), ord("F")):
            self.posture = "fallen"
        elif low in (ord("r"), ord("R")) and self.ordered_quad is not None:
            self.rotation_180 = not self.rotation_180
            self.left_pips, self.right_pips = self.right_pips, self.left_pips
            self.message = "Preview rotated 180 degrees; left/right pip values swapped."
        elif ord("0") <= low <= ord("6") and self.ordered_quad is not None:
            value = low - ord("0")
            if self.pip_entry_side == 0:
                self.left_pips = value
                self.pip_entry_side = 1
                self.message = f"LEFT={value}. Now type RIGHT pip count (0-6)."
            else:
                self.right_pips = value
                self.pip_entry_side = 0
                self.message = f"Pips={self.left_pips}-{value}. Press Enter to save, or type again to edit LEFT."
        elif low in (10, 13):
            self._save_annotation()
        elif low == ord("["):
            if self.index > 0:
                self.index -= 1
                self._load_current()
        elif low == ord("]"):
            if self.index + 1 < len(self.images):
                self.index += 1
                self._load_current()
        return True

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        running = True
        while running:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKeyEx(30)
            if key != -1:
                running = self.handle_key(key)
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        _atomic_write_json(self.annotation_path, self.data)
        cv2.destroyAllWindows()


def self_test() -> None:
    image = np.full((300, 600, 3), 230, dtype=np.uint8)
    points = [[50, 30], [560, 55], [540, 265], [70, 250]]
    ordered = order_quad(points)
    crop = rectify(image, ordered)
    assert crop.shape == (RECTIFIED_SIZE[1], RECTIFIED_SIZE[0], 3)
    assert len(np.unique(np.round(rotate_quad_180(ordered), 3), axis=0)) == 4
    print("annotation geometry self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    try:
        DominoAnnotator(args.images_dir, args.output_dir).run()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

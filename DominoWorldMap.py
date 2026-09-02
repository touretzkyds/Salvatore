from aim_fsm import *

import cv2
import numpy as np


class DominoWorldMap(StateMachineProgram):
    def __init__(self):
        super().__init__(
            launch_cam_viewer=True,
            launch_worldmap_viewer=True,
            launch_path_viewer=True,
            speech=False,
            aruco=False,
            domino=True,
            domino_labeling=True,
            force_annotation=True,
        )

    def user_image(self, image, gray):
        detector = getattr(self.robot, "domino_detector", None)
        if detector is not None:
            detector.detect(image, frame_id=getattr(self.robot, "frame_count", None))

    def user_annotate(self, image):
        out = image.copy()
        detector = getattr(self.robot, "domino_detector", None)
        debug_sheet = None
        if detector is not None:
            debug_sheet = detector.latest_debug_contact_sheet(columns=2)
            if debug_sheet is not None:
                imshow("domino-label-debug", debug_sheet)
            for obs in detector.latest_observations():
                quad = np.array(obs.quad, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [quad], True, (255, 120, 30), 1, lineType=cv2.LINE_AA)
                if getattr(obs, "divider_endpoints", None):
                    p0 = tuple(int(round(v)) for v in obs.divider_endpoints[0])
                    p1 = tuple(int(round(v)) for v in obs.divider_endpoints[1])
                    cv2.line(out, p0, p1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
                if getattr(obs, "half_image_centers", None) and getattr(obs, "half_counts", None):
                    for idx, ((hx, hy), count) in enumerate(zip(obs.half_image_centers, obs.half_counts)):
                        color = (255, 80, 80) if idx == 0 else (80, 180, 255)
                        cv2.circle(out, (int(round(hx)), int(round(hy))), 5, color, -1, lineType=cv2.LINE_AA)
                        cv2.putText(
                            out,
                            str(count),
                            (int(round(hx)) + 6, int(round(hy)) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            2,
                            lineType=cv2.LINE_AA,
                        )
        snapshot = self.robot.world_map.snapshot_objects()
        for (object_id, obj) in snapshot.items():
            if not isinstance(obj, DominoObj):
                continue
            if not getattr(obj, "image_quad", None):
                continue
            color = (0, 220, 90) if getattr(obj, "is_visible", False) else (140, 140, 140)
            quad = np.array(obj.image_quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [quad], True, color, 2, lineType=cv2.LINE_AA)
            if getattr(obj, "image_axis", None):
                p0 = tuple(int(round(v)) for v in obj.image_axis[0])
                p1 = tuple(int(round(v)) for v in obj.image_axis[1])
                cv2.line(out, p0, p1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            if getattr(obj, "image_divider", None):
                p0 = tuple(int(round(v)) for v in obj.image_divider[0])
                p1 = tuple(int(round(v)) for v in obj.image_divider[1])
                cv2.line(out, p0, p1, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            if getattr(obj, "image_center", None):
                cx, cy = obj.image_center
                label = object_id
                if getattr(obj, "face_label", None):
                    label += f" {obj.face_label}"
                cv2.putText(
                    out,
                    label,
                    (int(round(cx)) + 4, int(round(cy)) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
            if getattr(obj, "half_image_centers", None) and getattr(obj, "half_counts", None):
                for idx, ((hx, hy), count) in enumerate(zip(obj.half_image_centers, obj.half_counts)):
                    if count is None:
                        continue
                    color = (255, 80, 80) if idx == 0 else (80, 180, 255)
                    cv2.circle(out, (int(round(hx)), int(round(hy))), 4, color, -1, lineType=cv2.LINE_AA)
                    cv2.putText(
                        out,
                        str(count),
                        (int(round(hx)) + 5, int(round(hy)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        lineType=cv2.LINE_AA,
                    )
        return out

    class ReportDominoes(StateNode):
        def start(self, event=None):
            super().start(event)
            dominoes = [
                obj for obj in self.robot.world_map.objects.values()
                if isinstance(obj, DominoObj)
            ]
            if not dominoes:
                print("No dominoes in world map.")
            else:
                for obj in sorted(dominoes, key=lambda item: item.id or ""):
                    print(obj)
            self.post_completion()

    def setup(self):
        intro = Print("\nDominoWorldMap running.\nType 'tm' to print mapped dominoes.") .set_name("intro") .set_parent(self)
        loop = StateNode() .set_name("loop") .set_parent(self)
        report = self.ReportDominoes() .set_name("report") .set_parent(self)

        completiontrans1 = CompletionTrans() .set_name("completiontrans1")
        completiontrans1 .add_sources(intro) .add_destinations(loop)

        textmsgtrans1 = TextMsgTrans() .set_name("textmsgtrans1")
        textmsgtrans1 .add_sources(loop) .add_destinations(report)

        completiontrans2 = CompletionTrans() .set_name("completiontrans2")
        completiontrans2 .add_sources(report) .add_destinations(loop)

        return self

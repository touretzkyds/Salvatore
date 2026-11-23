"""Scan three standing dominoes and join two using the VEX AIM vest arms.

Task details:
Start on the right of a row of three standing tiles, facing across the row
toward an empty assembly area. The robot scans to its left. The initial layout
uses roughly 200 mm between tiles and a 250 mm approach distance. Measurements
and targets use startup odometry (millimetres, radians).
"""

import asyncio
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations
import math
import os
import time

import cv2
import numpy as np
import vex

from aim_fsm.domino import CNNDominoLabelProvider, directed_long_axis
from aim_fsm.domino_display import DominoDisplay
from aim_fsm.nodes import ActionNode
from aim_fsm.worldmap import GROUND_PROJECTION_K1, GROUND_PROJECTION_K2
from kickpickrefine import KNOWN_LENGTH, PROJECT_DIR, contour_quad, estimate_divider_x, kickpickrefine


# Set either filename to None to use kickpickrefine's original checkpoint.
STANDING_WEIGHTS_FILE = "more_data_standing.pt"
FALLEN_WEIGHTS_FILE = "more_data_fallen.pt"
SCAN_ONLY = os.environ.get("DOMINO_SCAN_ONLY") == "1"
KNOCK_ONLY = os.environ.get("DOMINO_KNOCK_ONLY") == "1"

TILE_LENGTH_MM = KNOWN_LENGTH * 10.0
SCAN_STEP_MM = 50.0
SCAN_STOPS = 13
STANDING_CONFIDENCE = 0.60
FALLEN_FACE_CONFIDENCE = 0.65
ROW_DEPTH_TOL_MM = 55.0
SCAN_TRACK_DEPTH_MM = 65.0
SCAN_TRACK_LATERAL_MM = 38.0
SCAN_DUPLICATE_LATERAL_MM = TILE_LENGTH_MM / 2
SCAN_CONFIRM_FRAMES = 3
STANDING_RECHECK_FRAMES = 5
STANDING_RECHECK_OFFSETS_MM = (25.0, -25.0)  # Change viewpoint without approaching the row.
STANDING_ASSIGNMENT_MARGIN_MM = 15.0
FALLEN_RECHECK_FRAMES = 3
FALLEN_KNOCK_POSE_RADIUS_MM = 35.0
FALLEN_TRACK_POSE_RADIUS_MM = 30.0
FALLEN_POSE_ANGLE_TOL = math.radians(40.0)
FALLEN_POSE_JITTER_MM = 12.0
FALLEN_POSE_JITTER_ANGLE = math.radians(15.0)
FALLEN_RECHECK_BACK_MM = 20.0
FALLEN_RECHECK_SIDE_MM = 35.0
REQUIRE_DISTINCT_FACES = True  # The demo uses three tiles from one set.
KNOCK_STANDOFF_MM = 95.0
KNOCK_SHIFT_MM = 40.0
KNOCK_END_FORWARD_MM = 10.0  # Target's forward distance from chassis center at the end.
KNOCK_YAW_DEG = 30.0
KNOCK_RIGHT_MM = 24.0
KNOCK_LATERAL_TOL_MM = 8.0
KNOCK_MAX_YAW_DEG = 30.0
GUIDE_CENTER_OFFSET_MM = 65.0  # Base center to tile center while pushing. Measure on your rig.
TILE_FACE_HEIGHT_MM = 8.0  # Height of the flat tile's visible face above the table.
GUIDE_CONTACT_CLEARANCE_MM = 28.0
GUIDE_APPROACH_DISTANCES_MM = (GUIDE_CENTER_OFFSET_MM + GUIDE_CONTACT_CLEARANCE_MM, 85.0, 78.0, 105.0, 115.0)
GUIDE_TRANSIT_CLEARANCE_MM = GUIDE_CENTER_OFFSET_MM + 5.0
GUIDE_CAPTURE_MM = 25.0
GUIDE_CAPTURE_TOL_MM = 12.0
ASSEMBLY_AHEAD_MM = 620.0  # Far tile center, measured beyond the original standing row.
ROBOT_CLEARANCE_MM = 95.0  # Includes the external arms and a margin around other tiles.
PUSH_STEP_MM = 25.0
FINAL_PUSH_STEP_MM = 7.0
POSE_TOL_MM = 12.0
JOINT_GAP_MM = 5.0
JOINT_LATERAL_TOL_MM = 8.0
JOINT_ANGLE_TOL = math.radians(8.0)
DRIVE_SPEED = 90
KNOCK_SPEED = 60
PUSH_SPEED = 35
TURN_SPEED = 35
TURN_RADIUS_MM = 85.0
RESULT_DISPLAY_SECONDS = 3.0
CAMERA_SETTLE_SECONDS = 0.30
CAMERA_SKIP_FRAMES = 2
CAMERA_STABLE_POSITION_MM = 2.0
CAMERA_STABLE_HEADING_RAD = math.radians(1.0)


class DemoStopped(RuntimeError):
    def __init__(self, message: str, *, screen_reason: str = "CHECK LOG"):
        super().__init__(message)
        self.screen_reason = screen_reason


class TileStillStanding(DemoStopped):
    """A matching upright read was confirmed and no nearby flat tile was seen."""


@dataclass(frozen=True)
class Tile:
    x: float
    y: float
    heading: float  # Points from the first pip value toward the second.
    pips: tuple[int, int] | None
    divider_visible: bool = False
    pip_hints: tuple[int | None, int | None] | None = None

    def facing(self, pips):
        if self.pips == pips:
            return self
        if self.pips is None or self.pips[::-1] != pips:
            raise DemoStopped(f"Read {self.pips}, expected {pips}", screen_reason="PIP MISMATCH")
        return Tile(self.x, self.y, wrap(self.heading + math.pi), pips, self.divider_visible)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def axis_error(first, second):
    """Difference between two long axes without assuming which end leads."""
    return abs(wrap(2 * (first - second))) / 2


def compatible_flat_orders(read, pips):
    """A positive pip read can resolve an end; a blank cannot."""
    hints = read.pip_hints or read.pips or (None, None)
    return [reverse for reverse in (False, True)
            if all(value in (None, 0, expected)
                   for value, expected in zip(hints, pips[::-1] if reverse else pips))]


def orient_flat_read(read, pips, reverse):
    heading = wrap(read.heading + math.pi) if reverse else read.heading
    return Tile(read.x, read.y, heading, pips, read.divider_visible)


def offset(x, y, heading, distance):
    return x + distance * math.cos(heading), y + distance * math.sin(heading)


def displacement(x, y, heading, target_x, target_y):
    dx, dy = target_x - x, target_y - y
    return (dx * math.cos(heading) + dy * math.sin(heading),
            -dx * math.sin(heading) + dy * math.cos(heading))


def distance_to_segment(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    px, py = point[0] - start[0], point[1] - start[1]
    length_sq = dx * dx + dy * dy
    fraction = (px * dx + py * dy) / length_sq if length_sq else 0.0
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(px - fraction * dx, py - fraction * dy)


def distinct_standing_reads(tiles):
    """Input is ranked by confidence; keep one detection per physical footprint."""
    distinct = []
    for tile in tiles:
        if not any(abs(tile.y - seen.y) < SCAN_DUPLICATE_LATERAL_MM
                   and abs(tile.x - seen.x) < ROW_DEPTH_TOL_MM for seen in distinct):
            distinct.append(tile)
    return distinct


def scan_track_tile(track):
    reads = list(track.values())
    pips, _ = Counter(tile.pips for tile in reads).most_common(1)[0]
    return Tile(float(np.median([tile.x for tile in reads])),
                float(np.median([tile.y for tile in reads])), reads[-1].heading, pips)


def merge_split_scan_track(tracks):
    """Fuse nearby same-face tracks with no evidence of two simultaneous tiles."""
    stable = []
    for index, track in enumerate(tracks):
        faces = Counter(tuple(sorted(tile.pips)) for tile in track.values())
        face, count = faces.most_common(1)[0]
        if count >= SCAN_CONFIRM_FRAMES and count > len(track) / 2:
            stable.append((index, scan_track_tile(track), face))

    for (i, first, first_face), (j, second, second_face) in combinations(stable, 2):
        if (first_face == second_face
                and abs(first.x - second.x) <= ROW_DEPTH_TOL_MM
                and abs(first.y - second.y) < TILE_LENGTH_MM
                and tracks[i].keys().isdisjoint(tracks[j])):
            tracks[i] = dict(sorted((tracks[i] | tracks[j]).items()))
            del tracks[j]
            print(f"[DOMINO DEMO] Scan merged nearby {first_face} tracks at "
                  f"y={first.y:.0f} and y={second.y:.0f} mm")
            return True
    return False


def standing_assignment(slots, observations):
    """Match the whole row by geometry. Omit slots that can plausibly swap matches."""
    slots = sorted(slots, key=lambda tile: tile.y)
    observations = sorted(distinct_standing_reads(observations), key=lambda tile: tile.y)
    options = []

    def search(slot_index, next_read, matches, matched, cost):
        if slot_index == len(slots):
            options.append((-matched, cost, matches))
            return
        search(slot_index + 1, next_read, (*matches, None), matched, cost)
        slot = slots[slot_index]
        for index in range(next_read, len(observations)):
            read = observations[index]
            dx, dy = abs(read.x - slot.x), abs(read.y - slot.y)
            if dx <= SCAN_TRACK_DEPTH_MM and dy <= SCAN_TRACK_LATERAL_MM:
                # Ordering along the row identifies neighbors even when their gates overlap.
                search(slot_index + 1, index + 1, (*matches, index), matched + 1,
                       cost + dy + 0.25 * dx)

    search(0, 0, (), 0, 0.0)
    count, cost, best = min(options, key=lambda option: option[:2])
    uncertain = {slot for other_count, other_cost, other in options
                 if other_count == count and other_cost < cost + STANDING_ASSIGNMENT_MARGIN_MM
                 for slot, chosen, alternate in zip(slots, best, other) if chosen != alternate}
    return ({slot: observations[index] for slot, index in zip(slots, best)
             if index is not None and slot not in uncertain}, uncertain)


def matching_pairs(tiles):
    """Yield distinct tiles and the oriented pip order of each possible chain."""
    for far, near in permutations(sorted(tiles, key=lambda tile: tile.y, reverse=True), 2):
        for far_pips in (far.pips, far.pips[::-1]):
            for near_pips in (near.pips, near.pips[::-1]):
                if near_pips[1] == far_pips[0]:
                    yield far, near, far_pips, near_pips


def lane_change(lateral):
    """Two opposite arcs with a straight segment between them, in chain coordinates."""
    distance = abs(lateral)
    angle = min(math.pi / 4, math.acos(max(0.0, 1.0 - distance / (2 * TURN_RADIUS_MM))))
    if angle == 0:
        return 0.0, 0.0, 0.0
    diagonal = max(0.0, distance - 2 * TURN_RADIUS_MM * (1 - math.cos(angle))) / math.sin(angle)
    advance = 2 * TURN_RADIUS_MM * math.sin(angle) + diagonal * math.cos(angle)
    return math.copysign(angle, lateral), diagonal, advance


class kickpickrefine2(kickpickrefine):
    def __init__(self):
        overrides = {}
        for option, filename in (("standing_weights", STANDING_WEIGHTS_FILE),
                                 ("fallen_weights", FALLEN_WEIGHTS_FILE)):
            if filename is not None:
                path = os.path.join(PROJECT_DIR, "weights", filename)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Domino demo weights not found: {path}")
                overrides[option] = path
        super().__init__(speech=False, **overrides)
        path = os.path.join(PROJECT_DIR, "weights", "fallenhalf.pt")
        self.fallen_labels = CNNDominoLabelProvider(path) if os.path.isfile(path) else None

    def setup(self):
        self.Demo().set_name("join_two_dominoes").set_parent(self)
        return self

    class Demo(ActionNode):
        def __init__(self):
            super().__init__()
            self.task = None
            self.motion = None
            self.stopping = False
            self.row_obstacles = []
            self.placed_tiles = []
            self.last_read_summary = ""
            self.knock_shift = KNOCK_SHIFT_MM
            self.knock_end_forward = KNOCK_END_FORWARD_MM
            self.knock_yaw = math.radians(KNOCK_YAW_DEG)
            self.knock_right = KNOCK_RIGHT_MM
            self._vision_ready_pose = None

        def start(self, event=None):
            if self.running:
                return
            self.stopping = False
            super().start(event)
            self.task = self.robot.loop.create_task(self.run())

        def complete(self):
            if self.motion is not None and not self.motion.done():
                self.motion.set_result(None)

        def stop(self):
            self.stopping = True
            if self.task is not None and not self.task.done():
                self.robot.loop.call_soon_threadsafe(self.task.cancel)
            self.stop_drive()
            super().stop()

        def stop_drive(self):
            drive = self.robot.actuators.get("drive")
            if drive is not None and drive.holder is self:
                try:
                    drive.stop()
                finally:
                    drive.clear()

        def pose(self):
            sdk = self.robot.robot0
            return (float(sdk.get_y_position()), -float(sdk.get_x_position()),
                    wrap(-math.radians(sdk.inertial.get_heading())))

        async def drive(self, method, *args, timeout=8.0):
            if self.stopping:
                raise asyncio.CancelledError
            if getattr(self.robot, "was_picked_up", False):
                raise DemoStopped("Robot was picked up during the run", screen_reason="PICKED UP")
            actuator = self.robot.actuators["drive"]
            self._vision_ready_pose = None
            self.motion = self.robot.loop.create_future()
            try:
                getattr(actuator, method)(self, *args)
                await asyncio.wait_for(self.motion, timeout)
            except TimeoutError as exc:
                raise DemoStopped(f"Motion timed out during {method}", screen_reason="MOTION TIMEOUT") from exc
            finally:
                self.stop_drive()
                self.motion = None
                self._vision_ready_pose = None

        async def turn_to(self, heading):
            delta = wrap(heading - self.pose()[2])
            if abs(delta) > math.radians(2.0):
                await self.drive("turn", delta, TURN_SPEED)

        async def go_to(self, x, y, *, ignore=()):
            base_x, base_y, heading = self.pose()
            dx, dy = x - base_x, y - base_y
            distance = math.hypot(dx, dy)
            if distance < 4.0:
                return
            self.check_clearance(base_x, base_y, x, y, ignore=ignore)
            await self.drive("move_for", distance,
                             math.degrees(wrap(math.atan2(dy, dx) - heading)), DRIVE_SPEED,
                             timeout=distance / DRIVE_SPEED + 5.0)

        def check_clearance(self, x, y, target_x, target_y, ignore=()):
            for tile in self.row_obstacles + self.placed_tiles:
                if any(tile is item for item in ignore):
                    continue
                if distance_to_segment((tile.x, tile.y), (x, y), (target_x, target_y)) < ROBOT_CLEARANCE_MM:
                    raise DemoStopped(f"Route too close to tile {tile.pips}", screen_reason="PATH BLOCKED")

        def route(self, target_x, target_y, *, pickup=None):
            """Find a short route for the unloaded robot around the known tiles."""
            start = self.pose()[:2]
            goal = (target_x, target_y)
            if math.dist(start, goal) < 4.0:
                return []

            def clear(a, b):
                try:
                    self.check_clearance(*a, *b)
                except DemoStopped:
                    return False
                return (pickup is None or distance_to_segment((pickup.x, pickup.y), a, b)
                        >= GUIDE_TRANSIT_CLEARANCE_MM)

            if clear(start, goal):
                return [goal]

            # The outer points let the circular clearance envelope go around corners.
            nodes = [start, goal]
            for tile in self.row_obstacles + self.placed_tiles:
                for index in range(8):
                    nodes.append(offset(tile.x, tile.y, index * math.pi / 4, ROBOT_CLEARANCE_MM + 18))
            if pickup is not None:
                for index in range(8):
                    nodes.append(offset(pickup.x, pickup.y, index * math.pi / 4, GUIDE_TRANSIT_CLEARANCE_MM + 18))
            distance = {0: 0.0}
            previous = {}
            pending = set(range(len(nodes)))
            while pending:
                current = min(pending, key=lambda index: distance.get(index, math.inf))
                if current not in distance:
                    break
                if current == 1:
                    path = []
                    while current:
                        path.append(nodes[current])
                        current = previous[current]
                    return path[::-1]
                pending.remove(current)
                for neighbor in pending:
                    length = distance[current] + math.dist(nodes[current], nodes[neighbor])
                    if length < distance.get(neighbor, math.inf) and clear(nodes[current], nodes[neighbor]):
                        distance[neighbor] = length
                        previous[neighbor] = current
            raise DemoStopped("No clear route around the remaining standing tiles", screen_reason="PATH BLOCKED")

        async def fresh_frame(self):
            # Flush frames after a stable stop once per motion, then let repeated
            # reads at the same pose use successive frames without another settle.
            stable_pose = self._vision_ready_pose
            settled = stable_pose is not None
            stable_since = None
            required_frame = self.robot.frame_count + 1
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if self.stopping:
                    raise asyncio.CancelledError
                now = time.monotonic()
                if self.motion is not None or not self.robot.robot0.is_stopped():
                    stable_pose = None
                    settled = False
                    self._vision_ready_pose = None
                else:
                    pose = self.pose()
                    if (stable_pose is None
                            or math.dist(pose[:2], stable_pose[:2]) > CAMERA_STABLE_POSITION_MM
                            or abs(wrap(pose[2] - stable_pose[2])) > CAMERA_STABLE_HEADING_RAD):
                        stable_pose = pose
                        stable_since = now
                        settled = False
                        self._vision_ready_pose = None
                    if not settled and now - stable_since >= CAMERA_SETTLE_SECONDS:
                        required_frame = self.robot.frame_count + CAMERA_SKIP_FRAMES + 1
                        settled = True
                    frame_id = self.robot.frame_count
                    frame = self.robot.camera_image
                    if (settled and frame_id >= required_frame and frame is not None
                            and frame_id == self.robot.frame_count):
                        self._vision_ready_pose = stable_pose
                        return frame.copy(), frame_id
                await asyncio.sleep(0.025)
            if not settled:
                raise DemoStopped("Robot did not settle for the camera read", screen_reason="WAIT FOR STOP")
            raise DemoStopped("No fresh camera frame", screen_reason="NO CAMERA FRAME")

        def ground(self, pixel, height=0.0):
            hit, _ = self.robot.world_map.project_image_point_to_world(*map(float, pixel))
            x, y = float(hit[0, 0]), float(hit[1, 0])
            if height:
                # Intersect the same camera ray with the top of the flat tile.
                camera = self.robot.kine
                scale = 1 - height / camera.camera_height
                camera_x = GROUND_PROJECTION_K1 * camera.camera_from_origin + GROUND_PROJECTION_K2
                x, y = camera_x + (x - camera_x) * scale, y * scale
            if not (math.isfinite(x) and math.isfinite(y) and 0.0 < x < 850.0):
                raise ValueError("Tile projects outside the work area")
            return x, y

        def world_point(self, point, pose):
            x, y, heading = pose
            forward, left = point
            return (x + forward * math.cos(heading) - left * math.sin(heading),
                    y + forward * math.sin(heading) + left * math.cos(heading))

        async def standing(self):
            frame, frame_id = await self.fresh_frame()
            pose = self.pose()
            diagnostics = {}
            candidates = self.parent.standing_candidates(frame, rectify_masks=True, diagnostics=diagnostics)
            self.parent.publish_dominoes(candidates, frame_id)
            tiles = []
            rejected = []
            for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
                if candidate["confidence"] < STANDING_CONFIDENCE:
                    rejected.append(f"{candidate['label']} confidence {candidate['confidence']:.2f}")
                    continue
                quad = candidate["quad"]
                # The bottom edge of the standing face touches the table.
                foot = np.mean(quad[np.argsort(quad[:, 1])[-2:]], axis=0)
                try:
                    x, y = self.world_point(self.ground(foot), pose)
                except ValueError as exc:
                    rejected.append(str(exc))
                    continue
                pips = tuple(map(int, candidate["label"].split("-")))
                tiles.append(Tile(x, y, wrap(pose[2] - math.pi / 2), pips))
            self.last_read_summary = (f"standing detections={diagnostics['detections']}, "
                                      f"pip reads={len(candidates)}, rejected={rejected}")
            return tiles

        async def scan_row(self):
            tracks = []
            for stop in range(SCAN_STOPS):
                if stop:
                    await self.drive("sideways", SCAN_STEP_MM, DRIVE_SPEED)
                for read_index in range(SCAN_CONFIRM_FRAMES):
                    sample = stop * SCAN_CONFIRM_FRAMES + read_index
                    frame_tiles = distinct_standing_reads(await self.standing())

                    matches = []
                    for track_index, track in enumerate(tracks):
                        center_x = float(np.median([tile.x for tile in track.values()]))
                        center_y = float(np.median([tile.y for tile in track.values()]))
                        for tile_index, tile in enumerate(frame_tiles):
                            depth = abs(tile.x - center_x) / SCAN_TRACK_DEPTH_MM
                            lateral = abs(tile.y - center_y) / SCAN_TRACK_LATERAL_MM
                            if depth <= 1 and lateral <= 1:
                                matches.append((math.hypot(depth, lateral), track_index, tile_index))
                    matched_tracks = set()
                    matched_tiles = set()
                    for _, track_index, tile_index in sorted(matches):
                        if track_index in matched_tracks or tile_index in matched_tiles:
                            continue
                        tracks[track_index][sample] = frame_tiles[tile_index]
                        matched_tracks.add(track_index)
                        matched_tiles.add(tile_index)
                    tracks.extend({sample: tile} for index, tile in enumerate(frame_tiles)
                                  if index not in matched_tiles)

                if REQUIRE_DISTINCT_FACES:
                    while merge_split_scan_track(tracks):
                        pass
                # A stable position with inconsistent pips can still be reread at its slot.
                stable = [scan_track_tile(track) for track in tracks if len(track) >= SCAN_CONFIRM_FRAMES]
                if len(stable) > 3:
                    seen = [(t.pips, round(t.x), round(t.y)) for t in stable]
                    raise DemoStopped(f"Confirmed {len(stable)} standing tiles: {seen}", screen_reason="EXTRA TILES")
                if len(stable) == 3:
                    if max(tile.x for tile in stable) - min(tile.x for tile in stable) > ROW_DEPTH_TOL_MM:
                        raise DemoStopped("The three standing tiles are not in one row", screen_reason="CHECK ROW")
                    print(f"[DOMINO DEMO] Scan tracks {[(t.pips, round(t.x), round(t.y)) for t in stable]}")
                    self.row_obstacles = stable
                    return await self.confirm_row(stable)
            raise DemoStopped(f"Confirmed {len(stable)} of three standing tiles from {len(tracks)} tracks",
                              screen_reason="MISSING TILES")

        async def confirm_standing(self, expected, *, stage):
            # Match position before looking at pips, even after moving the camera.
            votes = []
            viewpoints = []
            pip_views = {}
            for lateral in (0.0, *STANDING_RECHECK_OFFSETS_MM):
                if lateral:
                    await self.go_to(self.back_x, expected.y + lateral)
                    await self.turn_to(0.0)
                readings = Counter()
                for _ in range(STANDING_RECHECK_FRAMES):
                    seen = await self.standing()
                    assigned, uncertain = standing_assignment(self.row_obstacles, seen)
                    tile = assigned.get(expected)
                    if tile is not None:
                        votes.append(tile)
                        readings[str(tile.pips)] += 1
                        pip_views.setdefault(tile.pips, set()).add(lateral)
                        if (tile.pips == expected.pips and readings[str(tile.pips)] >= SCAN_CONFIRM_FRAMES
                                and sum(read.pips == tile.pips for read in votes) > len(votes) / 2):
                            break
                    elif seen:
                        offsets = [(t.pips, round(t.x - expected.x), round(t.y - expected.y)) for t in seen]
                        if expected in uncertain:
                            reason = "ambiguous row assignment"
                        elif assigned:
                            reason = "only other row slots matched"
                        else:
                            reason = "outside row slots"
                        readings[f"{reason} (pips, dx, dy)={offsets}"] += 1
                    else:
                        readings[self.last_read_summary or "no usable standing read"] += 1
                viewpoints.append(f"offset {lateral:+.0f} mm: {dict(readings)}")
                if votes:
                    pips, count = Counter(t.pips for t in votes).most_common(1)[0]
                    supported = pips == expected.pips or len(pip_views[pips]) >= 2
                    if supported and count >= SCAN_CONFIRM_FRAMES and count > len(votes) / 2:
                        agreeing = [t for t in votes if t.pips == pips]
                        tile = agreeing[-1]
                        if lateral:
                            print(f"[DOMINO DEMO] {stage}: confirmed at y={expected.y:.0f} mm after "
                                  f"a lateral recheck; {'; '.join(viewpoints)}")
                            await self.go_to(self.back_x, expected.y)
                        if pips != expected.pips:
                            print(f"[DOMINO DEMO] {stage}: reread {expected.pips} as {pips} at y={tile.y:.0f} mm")
                        return Tile(float(np.median([t.x for t in agreeing])),
                                    float(np.median([t.y for t in agreeing])), tile.heading, pips)
            details = "; ".join(viewpoints)
            raise DemoStopped(f"{stage}: no consistent reading at y={expected.y:.0f} mm; {details}",
                              screen_reason="CHECK PIPS" if votes else "LOST TILE")

        async def confirm_row(self, tracks):
            confirmed = []
            for tile in sorted(tracks, key=lambda t: t.y, reverse=True):
                await self.go_to(self.back_x, tile.y)
                await self.turn_to(0.0)
                confirmed.append(await self.confirm_standing(tile, stage="centered scan"))
            if max(tile.x for tile in confirmed) - min(tile.x for tile in confirmed) > ROW_DEPTH_TOL_MM:
                raise DemoStopped("Rechecked tiles are not in one row", screen_reason="CHECK ROW")
            seen = [(t.pips, round(t.x), round(t.y)) for t in confirmed]
            if REQUIRE_DISTINCT_FACES and len({tuple(sorted(t.pips)) for t in confirmed}) < 3:
                raise DemoStopped(f"Duplicate face after centered scan; recheck pips or tile set: {seen}",
                                  screen_reason="DUPLICATE TILE")
            print(f"[DOMINO DEMO] Standing tiles {seen}")
            return confirmed

        def flat_tile(self, frame, candidate, pose):
            contour = candidate.get("contour")
            if contour is None:
                contour = cv2.boxPoints(candidate["rect"])
            if len(contour) < 4:
                raise ValueError("Fallen mask outline has fewer than four points")
            quad = contour_quad(contour)
            local_contour = np.array([self.ground(pixel, TILE_FACE_HEIGHT_MM) for pixel in contour],
                                     dtype=np.float32)
            local_quad = np.array([self.ground(pixel, TILE_FACE_HEIGHT_MM) for pixel in quad], dtype=np.float32)
            # The rectangle bounding the mask in image space includes background.
            # Fit the actual outline after projecting it to the tile's top plane.
            footprint = cv2.minAreaRect(local_contour)
            center, size, _ = footprint
            length, width = sorted(size, reverse=True)
            if not (28.0 <= length <= 85.0 and 10.0 <= width <= 45.0):
                x, y = self.world_point(center, pose)
                raise ValueError(f"Implausible flat tile dimensions {length:.1f} x {width:.1f} mm "
                                 f"at ({x:.0f}, {y:.0f}) mm")
            theta, _ = directed_long_axis(cv2.boxPoints(footprint))
            along = np.array((math.cos(theta), math.sin(theta)))
            across = np.array((-along[1], along[0]))
            # Order the image corners using their table-plane positions. Perspective
            # can put a far-side corner closer to the wrong end in the image.
            order = np.argsort((local_quad - center) @ along)
            side = (local_quad - center) @ across
            first_end = sorted(order[:2], key=side.__getitem__, reverse=True)
            second_end = sorted(order[2:], key=side.__getitem__, reverse=True)
            corners = quad[[first_end[0], second_end[0], second_end[1], first_end[1]]]
            destination = np.array([[0, 0], [119, 0], [119, 59], [0, 59]], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(corners, destination)
            crop = cv2.warpPerspective(frame, transform, (120, 60))
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 255)
            crop_mask = cv2.warpPerspective(mask, transform, (120, 60), flags=cv2.INTER_NEAREST)
            crop[crop_mask == 0] = 255
            split = estimate_divider_x(crop, 0, 6, 120, 54)
            divider_visible = False
            if split is not None:
                column = int(round(split))
                gray = cv2.cvtColor(crop[6:54], cv2.COLOR_RGB2GRAY).astype(float)
                line = gray[:, column - 2:column + 3].min(axis=1)
                sides = np.minimum(gray[:, column - 14:column - 6].mean(axis=1),
                                   gray[:, column + 7:column + 15].mean(axis=1))
                divider_visible = bool(np.mean(sides - line >= 18) >= 0.65)
            halves = ((crop[:, :column - 3], crop[:, column + 3:]) if divider_visible
                      else self.parent.split_halves(crop))
            labels = self.parent.fallen_labels
            reads = [labels.predict_half(labels.preprocess_half(half)) for half in halves]
            hints = tuple(value if confidence >= FALLEN_FACE_CONFIDENCE else None for value, confidence in reads)
            pips = None
            if all(value is not None for value in hints):
                pips = hints
                candidate["label"] = f"{pips[0]}-{pips[1]}"
                candidate["confidence"] = min(confidence for _, confidence in reads)
            x, y = self.world_point(center, pose)
            return Tile(x, y, wrap(pose[2] + theta), pips, divider_visible, hints)

        async def fallen(self):
            frame, frame_id = await self.fresh_frame()
            pose = self.pose()
            # Native masks use the camera frame's coordinates, including after YOLO letterboxing.
            results = self.parent.fallen_model.predict(
                source=frame, conf=0.25, iou=0.45, retina_masks=True, verbose=False,
            )
            candidates = []
            mask_count = 0
            if results and results[0].masks is not None:
                masks = results[0].masks.data
                height, width = frame.shape[:2]
                if masks.shape[-2:] != (height, width):
                    raise DemoStopped(
                        f"Fallen mask shape {tuple(masks.shape[-2:])} does not match frame {(height, width)}",
                        screen_reason="MASK SIZE ERROR",
                    )
                mask_count = len(masks)
                candidates = self.parent.fallen_candidates(frame, results[0], (1.0, 0, 0, width, height))
            tiles = []
            rejected = []
            for candidate in candidates:
                try:
                    tiles.append(self.flat_tile(frame, candidate, pose))
                except (ValueError, cv2.error) as exc:
                    rejected.append(str(exc))
            self.parent.publish_dominoes([c for c in candidates if "label" in c], frame_id, is_fallen=True)
            self.last_read_summary = f"fallen masks={mask_count}, candidates={len(candidates)}, rejected={rejected}"
            return tiles

        def nearby_flat_read(self, tiles, expected, radius):
            nearby = [tile for tile in tiles if math.hypot(tile.x - expected.x, tile.y - expected.y) <= radius]
            if len(nearby) != 1 or not compatible_flat_orders(nearby[0], expected.pips):
                return None
            tile = nearby[0]
            distance = math.hypot(tile.x - expected.x, tile.y - expected.y)
            for other in self.placed_tiles:
                if other.pips not in (expected.pips, expected.pips[::-1]):
                    if math.hypot(tile.x - other.x, tile.y - other.y) <= distance:
                        return None
            return tile

        def flat_pose_match(self, tiles, expected, radius):
            read = self.nearby_flat_read(tiles, expected, radius)
            if read is None:
                return None
            choices = [orient_flat_read(read, expected.pips, reverse)
                       for reverse in compatible_flat_orders(read, expected.pips)]
            tracked = min(choices, key=lambda value: abs(wrap(value.heading - expected.heading)))
            if abs(wrap(tracked.heading - expected.heading)) > FALLEN_POSE_ANGLE_TOL:
                return None
            return tracked

        @staticmethod
        def partial_flat_ends(reads, expected):
            directed = []
            for read in reads:
                orders = compatible_flat_orders(read, expected.pips)
                if len(orders) == 1:
                    directed.append(orient_flat_read(read, expected.pips, orders[0]))
            if not directed:
                return None, False
            latest = directed[-1]
            conflict = any(abs(wrap(read.heading - latest.heading)) > FALLEN_POSE_JITTER_ANGLE
                           for read in directed[:-1])
            return (None if conflict else latest), conflict

        def locate(self, tiles, expected, radius, *, stage="target tracking", allow_pose=True):
            matches = [tile for tile in tiles
                       if tile.pips in (expected.pips, expected.pips[::-1])
                       and math.hypot(tile.x - expected.x, tile.y - expected.y) <= radius]
            matches.sort(key=lambda tile: math.hypot(tile.x - expected.x, tile.y - expected.y))
            if len(matches) > 1:
                nearest = math.hypot(matches[0].x - expected.x, matches[0].y - expected.y)
                second = math.hypot(matches[1].x - expected.x, matches[1].y - expected.y)
                if second - nearest < 20.0:
                    raise DemoStopped(f"{stage}: two tiles could match {expected.pips}",
                                      screen_reason="AMBIGUOUS TILE")
            if matches:
                tile = matches[0].facing(expected.pips)
                if tile.pips[0] == tile.pips[1] and abs(wrap(tile.heading - expected.heading)) > math.pi / 2:
                    return Tile(tile.x, tile.y, wrap(tile.heading + math.pi), tile.pips)
                return tile
            if allow_pose:
                tracked = self.flat_pose_match(tiles, expected, min(radius, FALLEN_TRACK_POSE_RADIUS_MM))
                if tracked is not None:
                    return tracked
            seen = [(tile.pips, round(math.hypot(tile.x - expected.x, tile.y - expected.y)))
                    for tile in tiles]
            raise DemoStopped(
                f"{stage}: could not confirm {expected.pips} at ({expected.x:.0f}, {expected.y:.0f}) mm "
                f"within {radius:.0f} mm; seen (pips, distance mm)={seen}; {self.last_read_summary}",
                screen_reason="LOST TILE",
            )

        def knock_path(self, standing, *, ignore):
            start = (standing.x - KNOCK_STANDOFF_MM, standing.y)
            shifted = (start[0], standing.y + self.knock_shift)
            end = (standing.x - self.knock_end_forward, shifted[1] - self.knock_right)
            self.check_clearance(*start, *shifted, ignore=ignore)
            self.check_clearance(*shifted, *end, ignore=ignore)
            return start, shifted

        async def knock(self, tile):
            try:
                fallen = await self.knock_attempt(tile)
            except TileStillStanding:
                print(f"[DOMINO DEMO] {tile.pips} is still upright; rereading the target and retrying knock once")
                fallen = await self.knock_attempt(tile)
            self.row_obstacles.remove(tile)
            print(f"[DOMINO DEMO] Confirmed {fallen.pips} flat at ({fallen.x:.0f}, {fallen.y:.0f}) mm")
            return fallen

        async def knock_attempt(self, tile):
            await self.go_to(self.back_x, tile.y)
            await self.turn_to(0.0)
            standing = await self.confirm_standing(tile, stage="before knock (standing)")
            if standing.pips != tile.pips:
                raise DemoStopped(f"Target at y={standing.y:.0f} mm changed from {tile.pips} "
                                  f"to {standing.pips}; rescan before choosing the chain",
                                  screen_reason="PIP MISMATCH")
            start, shifted = self.knock_path(standing, ignore=(tile,))
            await self.go_to(*start, ignore=(tile,))
            await self.go_to(*shifted, ignore=(tile,))
            if self.knock_yaw:
                self.check_clearance(*self.pose()[:2], *shifted, ignore=(tile,))
                await self.turn_to(self.knock_yaw)
            x, y, heading = self.pose()
            if self.knock_yaw and abs(wrap(heading - self.knock_yaw)) > math.radians(3.0):
                raise DemoStopped(f"Knock heading outside tolerance: yaw={math.degrees(heading):.1f} degrees",
                                  screen_reason="OUT OF LINE")
            # A diagonal strike uses the row's axes independently of chassis yaw.
            angled_strike = bool(self.knock_yaw or self.knock_right)
            strike_heading = 0.0 if angled_strike else heading
            forward, left = displacement(x, y, strike_heading, standing.x, standing.y)
            travel = forward - self.knock_end_forward
            end_x, end_y = offset(x, y, strike_heading, travel)
            end = (end_x, end_y - self.knock_right)
            end_left = standing.y - end[1]
            if (not 0 < travel <= KNOCK_STANDOFF_MM - self.knock_end_forward + POSE_TOL_MM
                    or abs(left + self.knock_shift) > KNOCK_LATERAL_TOL_MM
                    or abs(end_left + self.knock_shift - self.knock_right) > KNOCK_LATERAL_TOL_MM):
                raise DemoStopped(f"Knock alignment outside tolerance: forward={forward:.0f}, left={left:.0f} mm",
                                  screen_reason="OUT OF LINE")
            self.check_clearance(x, y, *end, ignore=(tile,))
            dx, dy = end[0] - x, end[1] - y
            body_angle_deg = math.degrees(wrap(math.atan2(dy, dx) - heading)) if angled_strike else 0.0
            print(f"[DOMINO DEMO] Knock {standing.pips}: target relative to base along strike axes "
                  f"before strike (forward, left)=({forward:.0f}, {left:.0f}) mm; "
                  f"forward travel={travel:.0f} mm, right travel={self.knock_right:.0f} mm, "
                  f"requested end clearance={self.knock_end_forward:.0f} mm, "
                  f"chassis yaw={math.degrees(heading):+.1f} degrees (left positive), "
                  f"drive angle relative to chassis={body_angle_deg:+.1f} degrees (right negative), "
                  f"strike speed={KNOCK_SPEED} mm/s")
            if angled_strike:
                await self.drive("move_for", math.hypot(dx, dy), body_angle_deg, KNOCK_SPEED)
            else:
                await self.drive("forward", travel, KNOCK_SPEED)
            x, y, _ = self.pose()
            forward, left = displacement(x, y, strike_heading, standing.x, standing.y)
            print(f"[DOMINO DEMO] Knock {standing.pips}: relative to base along strike axes at end of strike "
                  f"(forward, left)=({forward:.0f}, {left:.0f}) mm (pre-strike tile position)")
            if angled_strike:
                await self.go_to(*shifted, ignore=(tile,))
                await self.turn_to(0.0)
            else:
                await self.drive("forward", -travel, DRIVE_SPEED)
            await self.go_to(*start, ignore=(tile,))
            return await self.confirm_fall(tile, standing)

        async def confirm_fall(self, tile, standing):
            failures = []
            pose_reads = []
            saw_nearby_flat = False
            for _ in range(FALLEN_RECHECK_FRAMES):
                seen = await self.fallen()
                saw_nearby_flat = saw_nearby_flat or any(
                    math.hypot(read.x - standing.x, read.y - standing.y) <= FALLEN_KNOCK_POSE_RADIUS_MM
                    for read in seen
                )
                try:
                    return self.locate(seen, standing, 95.0, stage="after knock (fallen)", allow_pose=False)
                except DemoStopped as exc:
                    failures.append(str(exc))
                nearby = self.nearby_flat_read(seen, standing, FALLEN_KNOCK_POSE_RADIUS_MM)
                if nearby is not None:
                    pose_reads.append(nearby)

            upright = []
            slots = [standing if other is tile else other for other in self.row_obstacles]
            for _ in range(SCAN_CONFIRM_FRAMES):
                assigned, _ = standing_assignment(slots, await self.standing())
                read = assigned.get(standing)
                if read is not None:
                    upright.append(read.pips)
            matching_upright = sum(pips in (standing.pips, standing.pips[::-1]) for pips in upright)
            status = ("Target still reads as standing" if matching_upright >= 2
                      else "Could not verify whether the target fell")
            reason = "STILL UPRIGHT" if matching_upright >= 2 else "CHECK FALL"
            stable = [(previous, current) for previous, current in combinations(pose_reads, 2)
                      if math.hypot(previous.x - current.x, previous.y - current.y) <= FALLEN_POSE_JITTER_MM
                      and axis_error(previous.heading, current.heading) <= FALLEN_POSE_JITTER_ANGLE]
            if stable and not matching_upright:
                flat = stable[-1][1]
                consistent = [read for read in pose_reads
                              if math.hypot(read.x - flat.x, read.y - flat.y) <= FALLEN_POSE_JITTER_MM
                              and axis_error(read.heading, flat.heading) <= FALLEN_POSE_JITTER_ANGLE]
                directed, conflict = self.partial_flat_ends(consistent, standing)
                if directed is not None:
                    print(f"[DOMINO DEMO] Used visible pips on the flat tile to orient {standing.pips}")
                    return directed
                if KNOCK_ONLY:
                    reverse = abs(wrap(flat.heading - standing.heading)) > math.pi / 2
                    return orient_flat_read(flat, standing.pips, reverse)
                if not conflict:
                    for previous, current in stable:
                        if not (previous.divider_visible and current.divider_visible):
                            continue
                        tracked = self.flat_pose_match([current], standing, FALLEN_KNOCK_POSE_RADIUS_MM)
                        if tracked is None or axis_error(previous.heading, standing.heading) > FALLEN_POSE_ANGLE_TOL:
                            continue
                        print(f"[DOMINO DEMO] Flat geometry confirmed in two frames; "
                              f"retaining the standing read {standing.pips}")
                        return tracked
                print(f"[DOMINO DEMO] Confirmed flat tile near ({flat.x:.0f}, {flat.y:.0f}) mm; "
                      f"rereading the face for {standing.pips}")
                reread = await self.recheck_flat_face(tile, standing, flat, consistent)
                if reread is not None:
                    return reread
                status = "Target is flat; could not verify a face-up divider or end order for assembly"
                reason = "CHECK FACE"
            details = " | ".join(dict.fromkeys(failures))
            axes = [(round(t.x), round(t.y), round(math.degrees(axis_error(t.heading, standing.heading))),
                     t.divider_visible, t.pip_hints or t.pips) for t in pose_reads]
            message = (f"{status}; post-strike standing reads={upright}; "
                       f"flat poses (x, y, axis change deg, divider, pip hints)={axes}; {details}")
            if matching_upright >= 2 and not saw_nearby_flat:
                raise TileStillStanding(message, screen_reason=reason)
            raise DemoStopped(message, screen_reason=reason)

        async def recheck_flat_face(self, tile, standing, flat, consistent):
            view_x = min(self.pose()[0], flat.x - KNOCK_STANDOFF_MM) - FALLEN_RECHECK_BACK_MM
            rereads = []
            for side in (FALLEN_RECHECK_SIDE_MM, -FALLEN_RECHECK_SIDE_MM):
                view_y = flat.y + side
                try:
                    self.check_clearance(*self.pose()[:2], view_x, view_y, ignore=(tile,))
                except DemoStopped as exc:
                    print(f"[DOMINO DEMO] Skipping flat recheck viewpoint: {exc}")
                    continue
                await self.go_to(view_x, view_y, ignore=(tile,))
                x, y, _ = self.pose()
                await self.turn_to(math.atan2(flat.y - y, flat.x - x))
                for _ in range(FALLEN_RECHECK_FRAMES):
                    seen = await self.fallen()
                    nearby = [read for read in seen if math.hypot(read.x - flat.x, read.y - flat.y)
                              <= FALLEN_POSE_JITTER_MM * 1.5]
                    if len(nearby) != 1 or axis_error(nearby[0].heading, flat.heading) > FALLEN_POSE_JITTER_ANGLE:
                        continue
                    read = nearby[0]
                    if read.pips in (standing.pips, standing.pips[::-1]):
                        return read.facing(standing.pips)
                    if self.nearby_flat_read(nearby, standing, FALLEN_KNOCK_POSE_RADIUS_MM) is not None:
                        rereads.append(read)
                directed, conflict = self.partial_flat_ends([*consistent, *rereads], standing)
                if directed is not None:
                    print(f"[DOMINO DEMO] Used visible pips after changing viewpoint to orient {standing.pips}")
                    return directed
                if not conflict:
                    with_divider = [self.flat_pose_match([read], standing, FALLEN_KNOCK_POSE_RADIUS_MM)
                                    for read in rereads if read.divider_visible]
                    aligned = [read for read in with_divider if read is not None]
                    if len(aligned) >= 2:
                        return aligned[-1]
            return None

        def guide_approach(self, tile):
            capture_end = offset(tile.x, tile.y, tile.heading, GUIDE_CAPTURE_MM - GUIDE_CENTER_OFFSET_MM)
            failures = []
            for distance in GUIDE_APPROACH_DISTANCES_MM:
                approach = offset(tile.x, tile.y, tile.heading, -distance)
                try:
                    self.check_clearance(*approach, *capture_end)
                    return self.route(*approach, pickup=tile), approach, distance
                except DemoStopped as exc:
                    failures.append(f"{distance:g} mm: {exc}")
            heading_deg = math.degrees(tile.heading)
            raise DemoStopped(f"No clear guide approach for {tile.pips} at ({tile.x:.0f}, {tile.y:.0f}) mm, "
                              f"heading {heading_deg:+.0f} degrees; {'; '.join(failures)}",
                              screen_reason="PATH BLOCKED")

        async def line_up(self, tile):
            # The pips determine which short end enters the guide first.
            path, approach, stand_off = self.guide_approach(tile)
            print(f"[DOMINO DEMO] Guide pickup {tile.pips}: approach at ({approach[0]:.0f}, {approach[1]:.0f}) mm, "
                  f"stand-off={stand_off:.0f} mm, heading={math.degrees(tile.heading):+.1f} degrees (world)")
            for waypoint in path:
                await self.go_to(*waypoint)
            await self.turn_to(tile.heading)
            observed = self.locate(await self.fallen(), tile, 42.0)
            x, y, heading = self.pose()
            _, lateral = displacement(x, y, heading, observed.x, observed.y)
            if abs(lateral) > 4.0:
                await self.go_to(*offset(x, y, heading + math.pi / 2, lateral))
            await self.turn_to(observed.heading)
            observed = self.locate(await self.fallen(), observed, 30.0)
            forward, lateral = displacement(*self.pose(), observed.x, observed.y)
            if (abs(lateral) > 8.0 or abs(wrap(observed.heading - self.pose()[2])) > math.radians(9.0)
                    or not GUIDE_CENTER_OFFSET_MM + 4.0 <= forward <= GUIDE_CENTER_OFFSET_MM + 60.0):
                raise DemoStopped(f"Guide alignment outside tolerance for {tile.pips}", screen_reason="OUT OF LINE")
            print(f"[DOMINO DEMO] Guide aligned for {observed.pips}: "
                  f"heading={math.degrees(observed.heading):+.1f} degrees (world), "
                  f"forward={forward:.0f} mm, left={lateral:.0f} mm")
            distance = forward - GUIDE_CENTER_OFFSET_MM + GUIDE_CAPTURE_MM
            x, y, heading = self.pose()
            self.check_clearance(x, y, *offset(x, y, heading, distance))
            await self.drive("forward", distance, PUSH_SPEED)
            predicted = offset(observed.x, observed.y, observed.heading, GUIDE_CAPTURE_MM)
            carried = self.locate(await self.fallen(),
                                  Tile(*predicted, observed.heading, observed.pips), GUIDE_CAPTURE_TOL_MM)
            self.check_guided(carried)
            return carried

        async def push(self, tile, travel, *, arc=None):
            x, y, heading = self.pose()
            if arc is None:
                self.check_clearance(x, y, *offset(x, y, heading, travel))
                await self.drive("forward", travel, PUSH_SPEED)
                px, py = offset(tile.x, tile.y, heading, travel)
                predicted = Tile(px, py, tile.heading, tile.pips)
            else:
                radius, angle = arc
                goal_heading = heading + angle
                goal_x = x + radius * (math.sin(goal_heading) - math.sin(heading))
                goal_y = y - radius * (math.cos(goal_heading) - math.cos(heading))
                middle_heading = heading + angle / 2
                middle_x = x + radius * (math.sin(middle_heading) - math.sin(heading))
                middle_y = y - radius * (math.cos(middle_heading) - math.cos(heading))
                self.check_clearance(x, y, middle_x, middle_y)
                self.check_clearance(middle_x, middle_y, goal_x, goal_y)
                # DriveActuator derives omega from angle * radius * speed sign.
                # A right arc uses a negative omega and a negative radius for forward motion.
                await self.drive("drive_arc", radius, angle, None, PUSH_SPEED / radius)
                px, py = offset(goal_x, goal_y, goal_heading, GUIDE_CENTER_OFFSET_MM)
                predicted = Tile(px, py, wrap(tile.heading + angle), tile.pips)
            observed = self.locate(await self.fallen(), predicted, GUIDE_CAPTURE_TOL_MM)
            self.check_guided(observed)
            return observed

        def check_guided(self, tile):
            forward, lateral = displacement(*self.pose(), tile.x, tile.y)
            if (abs(lateral) > GUIDE_CAPTURE_TOL_MM
                    or abs(forward - GUIDE_CENTER_OFFSET_MM) > GUIDE_CAPTURE_TOL_MM
                    or abs(wrap(tile.heading - self.pose()[2])) > math.radians(12.0)):
                raise DemoStopped(f"Tile {tile.pips} slipped out of the guide", screen_reason="TILE SLIPPED")

        async def steer(self, tile, heading):
            for _ in range(16):
                delta = wrap(heading - self.pose()[2])
                if abs(delta) <= math.radians(2.0):
                    return tile
                angle = math.copysign(min(abs(delta), math.radians(15.0)), delta)
                tile = await self.push(tile, 0.0, arc=(math.copysign(TURN_RADIUS_MM, angle), angle))
            raise DemoStopped("Could not align the guide with the assembly line", screen_reason="OUT OF LINE")

        async def move_onto_line(self, tile, target):
            tile = await self.steer(tile, target.heading)
            for _ in range(3):
                x, y, _ = self.pose()
                goal = offset(target.x, target.y, target.heading, -GUIDE_CENTER_OFFSET_MM)
                forward, lateral = displacement(x, y, target.heading, *goal)
                if abs(lateral) <= JOINT_LATERAL_TOL_MM / 2:
                    return tile
                angle, diagonal, advance = lane_change(lateral)
                if advance + 30.0 > forward:
                    raise DemoStopped("Not enough room to align before the joint", screen_reason="NO ALIGN SPACE")
                tile = await self.steer(tile, target.heading + angle)
                while diagonal >= 4.0:
                    step = min(diagonal, PUSH_STEP_MM)
                    tile = await self.push(tile, step)
                    diagonal -= step
                tile = await self.steer(tile, target.heading)
            raise DemoStopped("Tile drifted during lane changes", screen_reason="OUT OF LINE")

        async def deliver(self, tile, target, *, anchor=None):
            carried = await self.line_up(tile)
            carried = await self.move_onto_line(carried, target)
            for _ in range(40):
                forward, lateral = displacement(carried.x, carried.y, target.heading, target.x, target.y)
                if abs(lateral) > JOINT_LATERAL_TOL_MM:
                    carried = await self.move_onto_line(carried, target)
                    continue
                if -POSE_TOL_MM <= forward <= POSE_TOL_MM:
                    break
                if forward < 0:
                    raise DemoStopped(f"Pushed tile {tile.pips} past the assembly position",
                                      screen_reason="PAST TARGET")
                distance = min(PUSH_STEP_MM if forward > 80.0 else FINAL_PUSH_STEP_MM, forward)
                carried = await self.push(carried, distance)
            else:
                raise DemoStopped(f"Exceeded push steps for {tile.pips}", screen_reason="PUSH LIMIT")
            if anchor is not None:
                carried = await self.close_joint(carried, anchor)
            await self.drive("forward", -45.0, DRIVE_SPEED)
            final = await self.fallen()
            placed = self.locate(final, carried, 22.0)
            if anchor is not None:
                fixed = self.locate(final, anchor, 22.0)
                self.verify_joint(placed, fixed)
            elif (math.hypot(placed.x - target.x, placed.y - target.y) > 20.0
                  or abs(wrap(placed.heading - target.heading)) > JOINT_ANGLE_TOL):
                raise DemoStopped("The first tile moved or turned during release", screen_reason="TILE MOVED")
            print(f"[DOMINO DEMO] Placed {placed.pips} at ({placed.x:.0f}, {placed.y:.0f}) mm")
            return placed

        @staticmethod
        def verify_joint(near, far):
            along, lateral = displacement(near.x, near.y, near.heading, far.x, far.y)
            gap = along - TILE_LENGTH_MM
            if (near.pips[1] != far.pips[0] or abs(lateral) > JOINT_LATERAL_TOL_MM
                    or abs(wrap(far.heading - near.heading)) > JOINT_ANGLE_TOL
                    or not -5.0 <= gap <= 12.0):
                raise DemoStopped(f"Joint outside tolerance (gap {gap:.1f} mm, offset {lateral:.1f} mm)",
                                  screen_reason="CHECK JOINT")

        async def close_joint(self, near, far):
            for attempt in range(6):
                seen = await self.fallen()
                fixed = self.locate(seen, far, 22.0)
                near = self.locate(seen, near, 22.0)
                if math.hypot(fixed.x - far.x, fixed.y - far.y) > 8.0:
                    raise DemoStopped("The anchor tile moved during assembly", screen_reason="TILE MOVED")
                along, lateral = displacement(near.x, near.y, near.heading, fixed.x, fixed.y)
                if (abs(lateral) > JOINT_LATERAL_TOL_MM
                        or abs(wrap(fixed.heading - near.heading)) > JOINT_ANGLE_TOL):
                    raise DemoStopped("Tile ends are out of alignment", screen_reason="OUT OF LINE")
                gap = along - TILE_LENGTH_MM
                if gap <= JOINT_GAP_MM + 4.0:
                    self.verify_joint(near, fixed)
                    return near
                if attempt < 5:
                    near = await self.push(near, min(FINAL_PUSH_STEP_MM, gap - JOINT_GAP_MM))
            raise DemoStopped("Could not close the joint", screen_reason="CHECK JOINT")

        async def return_to_row(self, next_tile):
            await self.turn_to(0.0)
            for waypoint in self.route(self.back_x, next_tile.y):
                await self.go_to(*waypoint)

        def pick_knock_test_tile(self):
            tiles = sorted(self.row_obstacles, key=lambda tile: tile.y, reverse=True)
            preferred = list(dict.fromkeys(far for far, _, _, _ in matching_pairs(tiles)))
            preferred.extend(tile for tile in tiles if tile not in preferred)
            for tile in preferred:
                try:
                    self.knock_path(tile, ignore=(tile,))
                except DemoStopped:
                    continue
                return tile
            # The strike rechecks the target and reports any actual obstruction.
            return preferred[0]

        def plan_pair(self, tiles, row_x):
            fallback = None
            fallback_checks = -1
            for far, near, far_pips, near_pips in matching_pairs(tiles):
                anchor = Tile(row_x + ASSEMBLY_AHEAD_MM, (far.y + near.y) / 2, 0, far_pips)
                plan = (far, near, anchor, near_pips)
                checks = 0
                # Prefer plans that can complete more of the sequence from the estimated poses.
                try:
                    for tile, pips, removed in ((far, far_pips, (far,)), (near, near_pips, (far, near))):
                        self.knock_path(tile, ignore=removed)
                        checks += 1
                        entry = offset(tile.x, tile.y, tile.facing(pips).heading,
                                       -(GUIDE_CENTER_OFFSET_MM + GUIDE_CONTACT_CLEARANCE_MM))
                        self.check_clearance(*entry, *entry, ignore=removed)
                        checks += 1
                except DemoStopped:
                    if checks > fallback_checks:
                        fallback = plan
                        fallback_checks = checks
                else:
                    return plan
            if fallback is not None:
                print("[DOMINO DEMO] Trying the best matching pair; checking each move as the tiles are observed")
                return fallback
            raise DemoStopped("None of the three tiles share an end value", screen_reason="NO MATCH")

        async def run(self):
            display = DominoDisplay(getattr(self.robot.robot0, "screen", None))
            detail = None
            try:
                try:
                    await self.join_tiles(display)
                except Exception as exc:
                    print(f"[DOMINO DEMO] Stopped: {exc}")
                    self.post_failure(str(exc))
                    result, color = "STOPPED", vex.Color.RED
                    detail = exc.screen_reason if isinstance(exc, DemoStopped) else "CHECK LOG"
                else:
                    self.post_success()
                    result = "SCAN OK" if SCAN_ONLY else "KNOCK OK" if KNOCK_ONLY else "JOINED"
                    color = vex.Color.GREEN
                finally:
                    self.stop_drive()

                if display.show(result, detail=detail, color=color):
                    await asyncio.sleep(RESULT_DISPLAY_SECONDS)
            finally:
                display.idle()

        async def join_tiles(self, display):
            parent = self.parent
            required = (parent.segmenter, parent.classifier)
            if not SCAN_ONLY:
                required += (parent.fallen_model, parent.fallen_labels)
            if any(model is None for model in required):
                raise DemoStopped("Required standing or fallen models are missing", screen_reason="MISSING MODEL")
            if not SCAN_ONLY:
                try:
                    self.knock_shift = float(os.environ.get("DOMINO_KNOCK_SHIFT_MM", KNOCK_SHIFT_MM))
                    self.knock_end_forward = float(os.environ.get("DOMINO_KNOCK_END_FORWARD_MM", KNOCK_END_FORWARD_MM))
                    yaw_deg = float(os.environ.get("DOMINO_KNOCK_YAW_DEG", KNOCK_YAW_DEG))
                    self.knock_right = float(os.environ.get("DOMINO_KNOCK_RIGHT_MM", KNOCK_RIGHT_MM))
                except ValueError as exc:
                    raise DemoStopped("Knock calibration values must be numeric", screen_reason="CHECK CALIBRATION") from exc
                if (not math.isfinite(self.knock_shift) or self.knock_shift == 0
                        or not 0 < self.knock_end_forward < KNOCK_STANDOFF_MM):
                    raise DemoStopped(
                        "Knock shift must be finite and nonzero, and end distance must be between 0 and 95 mm",
                        screen_reason="CHECK CALIBRATION",
                    )
                if not -KNOCK_MAX_YAW_DEG <= yaw_deg <= KNOCK_MAX_YAW_DEG:
                    raise DemoStopped(f"Knock yaw must be within +/-{KNOCK_MAX_YAW_DEG:g} degrees",
                                      screen_reason="CHECK CALIBRATION")
                if not math.isfinite(self.knock_right) or self.knock_right < 0:
                    raise DemoStopped("Knock right travel must be finite and nonnegative",
                                      screen_reason="CHECK CALIBRATION")
                self.knock_yaw = math.radians(yaw_deg)
            self.back_x = self.pose()[0]
            display.show("SCAN ROW")
            self.row_obstacles = await self.scan_row()
            if SCAN_ONLY:
                print("[DOMINO DEMO] Scan only: three tiles confirmed")
                return
            if KNOCK_ONLY:
                target = self.pick_knock_test_tile()
                print(f"[DOMINO DEMO] Knock only: testing {target.pips}")
                display.show("KNOCK", tile=(1, 1), pips=target.pips)
                flat = await self.knock(target)
                print(f"[DOMINO DEMO] Knock only: confirmed {flat.pips} flat")
                return

            display.show("MATCH")
            row_x = max(tile.x for tile in self.row_obstacles)
            far, near, anchor, near_pips = self.plan_pair(self.row_obstacles, row_x)
            print(f"[DOMINO DEMO] Planned chain {near_pips} -> {anchor.pips}")

            display.show("KNOCK", tile=(1, 2), pips=anchor.pips)
            far_flat = (await self.knock(far)).facing(anchor.pips)
            display.show("PLACE", tile=(1, 2), pips=far_flat.pips)
            far_placed = await self.deliver(far_flat, anchor)
            self.placed_tiles.append(far_placed)

            display.show("RETURN ROW")
            await self.return_to_row(near)
            display.show("KNOCK", tile=(2, 2), pips=near_pips)
            near_flat = (await self.knock(near)).facing(near_pips)
            near_x, near_y = offset(far_placed.x, far_placed.y, far_placed.heading,
                                    -(TILE_LENGTH_MM + JOINT_GAP_MM))
            display.show("PLACE", tile=(2, 2), pips=near_flat.pips)
            await self.deliver(near_flat, Tile(near_x, near_y, far_placed.heading, near_pips), anchor=far_placed)
            print(f"[DOMINO DEMO] Joined {near_pips} -> {anchor.pips}")

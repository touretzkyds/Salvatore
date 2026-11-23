"""Publish an FSM's existing domino observations without running another detector."""

import copy
import math
from types import SimpleNamespace

import cv2
import numpy as np

from .utils import PoseEstimate
from .worldmap import DominoObj


def observations_from_candidates(candidates, frame_id, *, is_fallen=False):
    """Adapt kickpickrefine's standing/fallen results to the world-map protocol."""
    observations = []
    for candidate in candidates:
        quad = (cv2.boxPoints(candidate["rect"]) if is_fallen
                else np.asarray(candidate["quad"], dtype=np.float32))
        rect = cv2.minAreaRect(quad)
        (cx, cy), (width, height), angle = rect
        if is_fallen:
            theta = math.radians(angle + (90.0 if width < height else 0.0))
            dx = 0.5 * max(width, height) * math.cos(theta)
            dy = 0.5 * max(width, height) * math.sin(theta)
            axis = ((cx - dx, cy - dy), (cx + dx, cy + dy))
        else:
            # The lowest long edge is where the standing tile meets the table.
            bottom = sorted(quad, key=lambda p: p[1], reverse=True)[:2]
            axis = tuple(sorted(bottom, key=lambda p: p[0]))
        label = candidate.get("label")
        counts = None if label is None else tuple(int(n) for n in label.split("-"))
        observations.append(SimpleNamespace(
            center_xy=(float(cx), float(cy)),
            quad=tuple((float(x), float(y)) for x, y in quad),
            axis_endpoints=tuple((float(x), float(y)) for x, y in axis),
            face_label=label,
            face_confidence=candidate.get("confidence") if label is not None else None,
            half_counts=counts,
            confidence=float(candidate.get("confidence", 0.0)),
            mask_area=float(candidate.get("mask_area", 0.0)),
            is_fallen=is_fallen,
            observed_frame=frame_id,
        ))
    return observations


class DominoMapPublisher:
    """Cache world positions at observation time, never reproject an old image.

    publish() immediately adds FSM-confirmed detections, bypassing the generic
    six-update confirmation rule. Normal map updates reuse these world-space
    snapshots. Only publish() supplies new image observations or pip readings.
    """

    def __init__(self, robot):
        self.robot = robot
        self._observations = []
        self._snapshots = []
        with robot.world_map._lock:
            self._known_ids = {key for key, obj in robot.world_map.objects.items()
                               if isinstance(obj, DominoObj)}

    def publish(self, observations):
        world_map = self.robot.world_map
        observations = list(observations)
        with world_map._lock:
            candidates = list(world_map.domino_objects_from_observations(observations))
            remaining = [world_map.objects[key] for key in sorted(self._known_ids)
                         if key in world_map.objects]
            snapshots = []
            for candidate in candidates:
                matches = []
                for previous in remaining:
                    if (candidate.half_counts is not None and previous.half_counts is not None
                            and sorted(candidate.half_counts) != sorted(previous.half_counts)):
                        continue
                    if candidate.is_fallen != previous.is_fallen:
                        # A knocked tile can move and rotate; its pips are unknown
                        # unless the fallen detector actually reads them.
                        cost = ((candidate.pose.x - previous.pose.x) ** 2
                                + (candidate.pose.y - previous.pose.y) ** 2)
                        limit = 80.0 ** 2
                    else:
                        cost = world_map.association_cost(candidate, previous)
                        limit = min(30.0 ** 2, world_map.max_association_cost(type(candidate)))
                    if cost < limit:
                        matches.append((cost, previous))
                if matches:
                    previous = min(matches, key=lambda item: item[0])[1]
                    remaining.remove(previous)
                    candidate.id = previous.id
                else:
                    candidate.id = world_map.next_in_sequence("Domino")
                world_map.objects[candidate.id] = candidate
                self._known_ids.add(candidate.id)
                world_map.missing_objects[:] = [obj for obj in world_map.missing_objects
                                                if obj.id != candidate.id]
                snapshot = copy.copy(candidate)
                snapshot.pose = PoseEstimate(candidate.pose)
                snapshots.append(snapshot)
            for previous in remaining:
                previous.is_visible = False
                previous.is_missing = world_map.should_be_visible(previous)
                if previous.is_missing and previous not in world_map.missing_objects:
                    world_map.missing_objects.append(previous)
            self._observations = list(observations)
            self._snapshots = snapshots

    def world_map_candidates(self):
        """Return copies so map association cannot mutate the cached snapshots."""
        with self.robot.world_map._lock:
            candidates = []
            for snapshot in self._snapshots:
                if snapshot.id not in self.robot.world_map.objects:
                    continue
                candidate = copy.copy(snapshot)
                candidate.pose = PoseEstimate(snapshot.pose)
                candidate.matched = None
                candidates.append(candidate)
            return candidates

    def detect(self, image, frame_id=None):
        """Compatibility interface: return cached readings, without inference."""
        return self.latest_observations()

    def latest_observations(self):
        return list(self._observations)

    def close(self):
        self._observations = []
        self._snapshots = []

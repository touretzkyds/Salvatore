"""Offline regression tests for associating a camera read with a domino slot."""

import contextlib
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import cv2
import numpy as np

from aim_fsm import evbase
import kickpickrefine2 as demo
from kickpickrefine2 import DemoStopped, Tile, kickpickrefine2, standing_assignment


def tile(x, y, pips):
    return Tile(x, y, 0, pips)


class StationaryFrameTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 0.0
        self.robot = SimpleNamespace(
            frame_count=0, camera_image=np.zeros((4, 4, 3), dtype=np.uint8),
            robot0=SimpleNamespace(is_stopped=lambda: True),
        )
        with patch.object(evbase, "robot_for_loading", self.robot, create=True):
            self.node = kickpickrefine2.Demo()
        self.node.pose = lambda: (0, 0, 0)

    async def tick(self, seconds):
        self.now = round(self.now + seconds, 6)
        self.robot.frame_count += 1

    async def test_waits_for_motion_and_pose_settling_then_skips_two_frames(self):
        self.robot.robot0.is_stopped = lambda: self.now >= 0.15
        self.node.pose = lambda: (3 if self.now >= 0.30 else 0, 0,
                                 math.radians(2) if self.now >= 0.40 else 0)
        sleep = AsyncMock(side_effect=self.tick)
        clock = SimpleNamespace(monotonic=lambda: self.now)
        with patch.object(demo, "time", clock), patch.object(demo.asyncio, "sleep", sleep):
            frame, frame_id = await self.node.fresh_frame()
            # Pose last moved at 0.40 s, followed by 0.30 s settling and three new frames.
            self.assertGreaterEqual(self.now, 0.775)
            self.assertEqual(frame_id, self.robot.frame_count)
            self.assertIsNot(frame, self.robot.camera_image)
            first_read_time = self.now
            _, next_frame_id = await self.node.fresh_frame()
            self.assertGreater(next_frame_id, frame_id)
            self.assertLess(self.now - first_read_time, 0.10)

    async def test_continuous_motion_times_out_without_accepting_frames(self):
        self.robot.robot0.is_stopped = lambda: False
        clock = SimpleNamespace(monotonic=lambda: self.now)
        with patch.object(demo, "time", clock), patch.object(demo.asyncio, "sleep", AsyncMock(side_effect=self.tick)):
            with self.assertRaisesRegex(DemoStopped, "did not settle"):
                await self.node.fresh_frame()
        self.assertIsNone(self.node._vision_ready_pose)

    async def test_camera_outage_after_settling_reports_missing_frame(self):
        async def no_image(seconds):
            self.now += seconds

        clock = SimpleNamespace(monotonic=lambda: self.now)
        with patch.object(demo, "time", clock), patch.object(demo.asyncio, "sleep", AsyncMock(side_effect=no_image)):
            with self.assertRaisesRegex(DemoStopped, "No fresh camera frame"):
                await self.node.fresh_frame()


class StandingRowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.right = tile(164, 14, (2, 3))
        self.middle = tile(168, 79, (3, 3))
        self.left = tile(170, 129, (5, 2))
        with patch.object(evbase, "robot_for_loading", SimpleNamespace(), create=True):
            self.node = kickpickrefine2.Demo()
        self.node.row_obstacles = [self.right, self.middle, self.left]
        self.node.back_x = 0
        self.node.go_to = AsyncMock()
        self.node.turn_to = AsyncMock()

    async def test_centered_confirmation_replays_overlapping_slots_from_log(self):
        self.node.standing = AsyncMock(side_effect=[
            [tile(165, 98, (3, 3)), tile(173, 150, (5, 2)), tile(159, 44, (0, 0))],
            [tile(165, 100, (3, 3)), tile(166, 150, (5, 2)), tile(162, 45, (0, 2))],
            [tile(165, 100, (3, 3)), tile(162, 45, (0, 2)), tile(166, 150, (5, 2))],
        ])
        read = await self.node.confirm_standing(self.left, stage="centered scan")
        self.assertEqual(read.pips, (5, 2))
        self.assertEqual(read.y, 150)
        self.node.go_to.assert_not_awaited()

    def test_oblique_view_matches_by_row_even_when_middle_is_closer_to_left_slot(self):
        middle = tile(163, 108, (3, 3))
        left = tile(173, 159, (5, 2))
        matches, uncertain = standing_assignment(self.node.row_obstacles, [left, middle])
        self.assertFalse(uncertain)
        self.assertEqual(matches, {self.middle: middle, self.left: left})

    def test_single_observation_near_slot_boundary_cannot_vote_for_either(self):
        matches, uncertain = standing_assignment(self.node.row_obstacles, [tile(163, 108, (5, 2))])
        self.assertFalse(matches)
        self.assertEqual(uncertain, {self.middle, self.left})

    async def test_missing_target_does_not_take_neighbor_votes(self):
        self.node.standing = AsyncMock(return_value=[tile(169, 92, (3, 3))])
        with self.assertRaisesRegex(DemoStopped, "only other row slots matched"):
            await self.node.confirm_standing(self.left, stage="centered scan")

    async def test_correction_uses_two_views(self):
        self.node.standing = AsyncMock(return_value=[tile(170, 150, (4, 2)), tile(165, 100, (3, 3))])
        with contextlib.redirect_stdout(io.StringIO()):
            read = await self.node.confirm_standing(self.left, stage="centered scan")
        self.assertEqual(read.pips, (4, 2))
        self.assertEqual(self.node.go_to.await_count, 2)

    async def test_split_scan_track_does_not_end_the_sweep_early(self):
        first = tile(154, 6, (3, 3))
        middle = tile(167, 170, (2, 3))
        split = tile(169, 209, (3, 2))  # Same face, projected 39 mm farther left.
        third = tile(168, 335, (5, 2))
        self.node.standing = AsyncMock(side_effect=[
            *([first, middle] for _ in range(3)),
            *([split] for _ in range(3)),
            *([third] for _ in range(3)),
        ])
        self.node.drive = AsyncMock()
        self.node.confirm_row = AsyncMock(side_effect=lambda tracks: tracks)
        with contextlib.redirect_stdout(io.StringIO()):
            tracks = await self.node.scan_row()
        self.assertEqual(self.node.drive.await_count, 2)
        self.assertEqual(len(tracks), 3)
        self.assertEqual({tuple(sorted(t.pips)) for t in tracks}, {(3, 3), (2, 3), (2, 5)})
        self.assertIn(third, tracks)
        self.node.confirm_row.assert_awaited_once()

    async def test_two_nearby_same_face_reads_seen_together_are_kept_for_recheck(self):
        first = tile(154, 6, (3, 3))
        middle = tile(167, 170, (2, 3))
        left = tile(169, 209, (2, 3))
        self.node.standing = AsyncMock(return_value=[first, middle, left])
        self.node.drive = AsyncMock()
        self.node.confirm_row = AsyncMock(side_effect=lambda tracks: tracks)
        with contextlib.redirect_stdout(io.StringIO()):
            tracks = await self.node.scan_row()
        self.assertEqual(tracks, [first, middle, left])
        self.node.drive.assert_not_awaited()


class KnockTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with patch.object(evbase, "robot_for_loading", SimpleNamespace(), create=True):
            self.node = kickpickrefine2.Demo()
        # Geometry tests set the required strike configuration explicitly.
        self.node.knock_end_forward = 30
        self.node.knock_yaw = 0
        self.node.knock_right = 0
        self.target = Tile(206, 49, -math.pi / 2, (3, 2))
        self.middle = Tile(200, 208, -math.pi / 2, (3, 3))
        self.left = Tile(187, 367, -math.pi / 2, (5, 2))
        self.node.row_obstacles = [self.target, self.middle, self.left]
        self.node.back_x = 0
        self.position = (0.0, 0.0, 0.0)
        self.node.pose = lambda: self.position

    async def test_default_calibration_is_used_when_no_environment_overrides_exist(self):
        with patch.object(evbase, "robot_for_loading", SimpleNamespace(), create=True):
            node = kickpickrefine2.Demo()
        self.assertEqual((node.knock_shift, node.knock_end_forward, node.knock_right), (40, 10, 24))
        self.assertAlmostEqual(math.degrees(node.knock_yaw), 30)
        node.pose = lambda: (0, 0, 0)
        node.parent = SimpleNamespace(segmenter=True, classifier=True, fallen_model=True, fallen_labels=True)
        node.scan_row = AsyncMock(return_value=self.node.row_obstacles)
        node.knock = AsyncMock(return_value=self.target)
        calibration_keys = {"DOMINO_KNOCK_SHIFT_MM", "DOMINO_KNOCK_END_FORWARD_MM",
                            "DOMINO_KNOCK_YAW_DEG", "DOMINO_KNOCK_RIGHT_MM"}
        environment = {key: value for key, value in demo.os.environ.items() if key not in calibration_keys}
        with patch.dict(demo.os.environ, environment, clear=True):
            with patch.object(demo, "SCAN_ONLY", False), patch.object(demo, "KNOCK_ONLY", True):
                with contextlib.redirect_stdout(io.StringIO()):
                    await node.join_tiles(Mock())
        self.assertEqual((node.knock_shift, node.knock_end_forward, node.knock_right), (40, 10, 24))
        self.assertAlmostEqual(math.degrees(node.knock_yaw), 30)

    def test_guide_approach_uses_shorter_pickup_distance_for_logged_row(self):
        self.position = (67, 341, 0)
        self.node.row_obstacles = [Tile(166, 152, -math.pi / 2, (2, 3)),
                                   Tile(185, -19, -math.pi / 2, (3, 3))]
        fallen = Tile(161, 328, math.pi / 2, (2, 5))
        with self.assertRaisesRegex(DemoStopped, "No clear route"):
            self.node.route(161, 235)

        path, approach, stand_off = self.node.guide_approach(fallen)
        self.assertEqual(stand_off, 78)
        self.assertEqual(approach, (161, 250))
        self.assertEqual(path[-1], approach)
        start = self.position[:2]
        self.assertLess(demo.distance_to_segment((fallen.x, fallen.y), start, approach),
                        demo.GUIDE_TRANSIT_CLEARANCE_MM)
        for waypoint in path:
            self.assertGreaterEqual(demo.distance_to_segment((fallen.x, fallen.y), start, waypoint),
                                    demo.GUIDE_TRANSIT_CLEARANCE_MM)
            start = waypoint

    async def test_knock_only_accepts_two_consistent_flat_reads_despite_blank_labels(self):
        self.node.fallen = AsyncMock(side_effect=[
            [Tile(222, 49, self.target.heading + math.pi, (0, 0))],
            [],
            [Tile(223, 51, self.target.heading, None)],
        ])
        self.node.standing = AsyncMock(side_effect=[[tile(207, 49, (2, 2))], [], []])
        with patch.object(demo, "KNOCK_ONLY", True), contextlib.redirect_stdout(io.StringIO()):
            result = await self.node.confirm_fall(self.target, self.target)
        self.assertEqual(result.pips, self.target.pips)
        self.assertEqual((result.x, result.y, result.heading), (223, 51, self.target.heading))

    async def test_full_demo_requires_a_visible_face_before_geometry_handoff(self):
        blank = Tile(222, 49, self.target.heading, (0, 0))
        self.node.standing = AsyncMock(return_value=[])
        self.node.fallen = AsyncMock(return_value=[blank])
        self.node.go_to = AsyncMock()
        self.node.turn_to = AsyncMock()
        with patch.object(demo, "KNOCK_ONLY", False):
            with self.assertRaisesRegex(DemoStopped, "could not verify a face-up divider"):
                await self.node.confirm_fall(self.target, self.target)
            self.node.fallen.return_value = [Tile(blank.x, blank.y, blank.heading, blank.pips, divider_visible=True)]
            with contextlib.redirect_stdout(io.StringIO()):
                result = await self.node.confirm_fall(self.target, self.target)
        self.assertEqual(result.pips, self.target.pips)
        reread = self.node.locate([blank], result, 12)
        self.assertEqual(reread.pips, self.target.pips)

    async def test_flat_axis_flip_still_reaches_face_recheck_and_recovers_end_order(self):
        first_axis = math.radians(2)
        second_axis = math.radians(-2)
        self.node.fallen = AsyncMock(side_effect=[
            [Tile(222, 49, first_axis, (0, 0))],
            [Tile(223, 49, second_axis, None)],
            [],
            [Tile(223, 49, second_axis, self.target.pips[::-1], divider_visible=True)],
        ])
        self.node.standing = AsyncMock(return_value=[])
        self.node.go_to = AsyncMock()
        self.node.turn_to = AsyncMock()
        with patch.object(demo, "KNOCK_ONLY", False), contextlib.redirect_stdout(io.StringIO()):
            result = await self.node.confirm_fall(self.target, self.target)
        self.node.go_to.assert_awaited_once()
        self.node.turn_to.assert_awaited_once()
        self.assertEqual(result.pips, self.target.pips)
        self.assertAlmostEqual(math.degrees(result.heading), 178)

    async def test_zero_two_read_orients_known_five_two_after_large_rotation(self):
        standing = Tile(139, 338, -math.pi / 2, (5, 2))
        heading = math.radians(-11)
        unknown = Tile(148, 332, heading, None)
        partial = Tile(148, 332, heading, (0, 2))
        self.node.row_obstacles = [standing, self.middle, self.target]
        self.node.fallen = AsyncMock(side_effect=[[partial], [unknown], [unknown]])
        self.node.standing = AsyncMock(return_value=[])
        self.node.go_to = AsyncMock()
        with patch.object(demo, "KNOCK_ONLY", False), contextlib.redirect_stdout(io.StringIO()):
            result = await self.node.confirm_fall(standing, standing)
        self.node.go_to.assert_not_awaited()
        self.assertEqual(result, Tile(148, 332, heading, (5, 2)))
        from_other_end = Tile(149, 333, heading + math.pi, None, pip_hints=(2, None))
        reread = self.node.locate([from_other_end], result, 12)
        self.assertEqual(reread.pips, (5, 2))
        self.assertAlmostEqual(reread.heading, heading)

    async def test_conflicting_partial_end_reads_do_not_select_a_pickup_direction(self):
        standing = Tile(139, 338, -math.pi / 2, (5, 2))
        forward = Tile(148, 332, math.radians(-11), (0, 2))
        reverse = Tile(148, 332, math.radians(-11), (2, 0))
        self.node.row_obstacles = [standing, self.middle, self.target]
        self.node.fallen = AsyncMock(side_effect=[[forward], [reverse], [], *([[]] * 6)])
        self.node.standing = AsyncMock(return_value=[])
        self.node.go_to = AsyncMock()
        self.node.turn_to = AsyncMock()
        with patch.object(demo, "KNOCK_ONLY", False), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(DemoStopped, "could not verify a face-up divider or end order"):
                await self.node.confirm_fall(standing, standing)

    async def test_flat_fallback_does_not_override_an_upright_match(self):
        self.node.fallen = AsyncMock(return_value=[Tile(222, 49, self.target.heading, (0, 0))])
        self.node.standing = AsyncMock(return_value=[self.target])
        with patch.object(demo, "KNOCK_ONLY", True):
            with self.assertRaisesRegex(DemoStopped, "Target still reads as standing") as stopped:
                await self.node.confirm_fall(self.target, self.target)
        self.assertNotIsInstance(stopped.exception, demo.TileStillStanding)

    def test_flat_mask_retains_geometry_and_detects_divider_when_pip_read_fails(self):
        frame = np.full((130, 160, 3), 255, dtype=np.uint8)
        contour = np.array([[30, 40], [126, 40], [126, 88], [30, 88]], dtype=np.int32)
        candidate = {"contour": contour, "rect": cv2.minAreaRect(contour)}
        self.node.ground = lambda point, height: np.asarray(point) / 2
        self.node.parent = SimpleNamespace(
            fallen_labels=SimpleNamespace(preprocess_half=lambda half: half, predict_half=lambda _: (None, 0)),
            split_halves=lambda crop: (crop[:, :60], crop[:, 60:]),
        )
        blank = self.node.flat_tile(frame, candidate, (0, 0, 0))
        self.assertIsNone(blank.pips)
        self.assertFalse(blank.divider_visible)
        cv2.line(frame, (78, 43), (78, 85), (0, 0, 0), 4)
        face = self.node.flat_tile(frame, candidate, (0, 0, 0))
        self.assertIsNone(face.pips)
        self.assertTrue(face.divider_visible)
        self.assertEqual((face.x, face.y), (39, 32))
        reads = iter([(0, 0.4), (2, 0.9)])
        self.node.parent.fallen_labels.predict_half = lambda _: next(reads)
        partial = self.node.flat_tile(frame, candidate, (0, 0, 0))
        self.assertIsNone(partial.pips)
        self.assertEqual(partial.pip_hints, (None, 2))

    async def test_strike_uses_calibrated_endpoint_and_retries_at_most_once_when_still_upright(self):
        async def go_to(x, y, **kwargs):
            # After the left shift the odometry places the robot 4 mm ahead of its waypoint.
            drift = 4 if y == self.target.y + 46 else 0
            self.position = (x + drift, y, 0.0)

        async def move(method, distance, speed):
            self.assertEqual(method, "forward")
            self.position = (self.position[0] + distance, self.position[1], 0.0)

        self.node.go_to = AsyncMock(side_effect=go_to)
        self.node.drive = AsyncMock(side_effect=move)
        self.node.confirm_standing = AsyncMock(return_value=self.target)
        self.node.fallen = AsyncMock(return_value=[])
        self.node.standing = AsyncMock(return_value=[self.target])
        self.node.knock_shift = 46
        self.node.knock_end_forward = 30
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(DemoStopped, "Target still reads as standing"):
                await self.node.knock(self.target)
        self.assertEqual([call.args[1] for call in self.node.drive.await_args_list], [61, -61, 61, -61])
        self.assertEqual(self.node.confirm_standing.await_count, 2)
        self.assertIn(self.target, self.node.row_obstacles)

    async def test_successful_retry_uses_new_target_position_and_removes_upright_obstacle(self):
        async def go_to(x, y, **kwargs):
            self.position = (x, y, self.position[2])

        async def move(method, distance, speed):
            self.assertEqual(method, "forward")
            self.position = (self.position[0] + distance, self.position[1], 0.0)

        moved = Tile(213, 52, self.target.heading, self.target.pips)
        flat = Tile(230, 51, self.target.heading, self.target.pips)
        self.node.go_to = AsyncMock(side_effect=go_to)
        self.node.drive = AsyncMock(side_effect=move)
        self.node.confirm_standing = AsyncMock(side_effect=[self.target, moved])
        self.node.standing = AsyncMock(return_value=[moved])
        self.node.fallen = AsyncMock(side_effect=[[], [], [], [flat]])
        with contextlib.redirect_stdout(io.StringIO()):
            result = await self.node.knock(self.target)
        self.node.go_to.assert_any_await(118, 52, ignore=(self.target,))
        self.assertEqual(result, flat)
        self.assertNotIn(self.target, self.node.row_obstacles)

    async def test_uncertain_fall_does_not_repeat_strike(self):
        self.node.knock_attempt = AsyncMock(side_effect=DemoStopped("Could not verify whether the target fell"))
        with self.assertRaisesRegex(DemoStopped, "Could not verify whether"):
            await self.node.knock(self.target)
        self.node.knock_attempt.assert_awaited_once_with(self.target)

    async def test_diagonal_strike_reaches_configured_endpoint_and_restores_camera_heading(self):
        async def go_to(x, y, **kwargs):
            self.position = (x, y, self.position[2])

        strike_pose = None

        async def move(method, *args):
            nonlocal strike_pose
            x, y, heading = self.position
            if method == "turn":
                self.position = (x, y, heading + args[0])
                return
            self.assertEqual(method, "move_for")
            distance, direction, _ = args
            world_direction = heading + math.radians(direction)
            self.position = (*demo.offset(x, y, world_direction, distance), heading)
            strike_pose = self.position

        async def fallen():
            self.assertAlmostEqual(self.position[2], 0)
            return [self.target]

        self.node.go_to = AsyncMock(side_effect=go_to)
        self.node.drive = AsyncMock(side_effect=move)
        self.node.confirm_standing = AsyncMock(return_value=self.target)
        self.node.fallen = AsyncMock(side_effect=fallen)
        self.node.knock_shift = 40
        self.node.knock_end_forward = 20
        self.node.knock_yaw = math.radians(30)
        self.node.knock_right = 21
        with contextlib.redirect_stdout(io.StringIO()):
            await self.node.knock(self.target)
        self.assertIsNotNone(strike_pose)
        self.assertAlmostEqual(strike_pose[0], self.target.x - 20)
        self.assertAlmostEqual(strike_pose[1], self.target.y + 40 - 21)
        self.assertAlmostEqual(math.degrees(strike_pose[2]), 30)
        command = next(call for call in self.node.drive.await_args_list if call.args[0] == "move_for")
        self.assertAlmostEqual(command.args[2], -45.5, delta=1.0)
        self.assertEqual(command.args[3], 60)

    def test_rejects_diagonal_that_brings_robot_too_close_to_neighbor(self):
        self.node.knock_shift = 40
        neighbor = Tile(self.target.x - 15, self.target.y + 40 - 100, 0, (5, 4))
        self.node.row_obstacles = [self.target, neighbor]
        self.node.knock_path(self.target, ignore=(self.target,))
        self.node.knock_right = 8
        with self.assertRaisesRegex(DemoStopped, "Route too close to tile"):
            self.node.knock_path(self.target, ignore=(self.target,))

    def test_closer_diagonal_endpoint_is_included_in_clearance_check(self):
        self.node.knock_shift = 40
        self.node.knock_right = 18
        neighbor = Tile(272, 45, 0, (5, 4))
        self.node.row_obstacles = [self.target, neighbor]
        self.node.knock_path(self.target, ignore=(self.target,))
        self.node.knock_end_forward = 20
        self.node.knock_right = 21
        with self.assertRaisesRegex(DemoStopped, "Route too close to tile"):
            self.node.knock_path(self.target, ignore=(self.target,))

    async def test_knock_only_runs_when_assembly_has_no_clear_plan(self):
        target = Tile(200, 0, -math.pi / 2, (3, 2))
        neighbors = [Tile(215, 80, -math.pi / 2, (3, 3)), Tile(200, 160, -math.pi / 2, (5, 2))]
        self.node.row_obstacles = [target, *neighbors]
        self.node.parent = SimpleNamespace(segmenter=True, classifier=True, fallen_model=True, fallen_labels=True)
        self.node.scan_row = AsyncMock(return_value=self.node.row_obstacles)
        self.node.knock = AsyncMock(return_value=target)
        self.node.deliver = AsyncMock()
        with patch.object(demo, "SCAN_ONLY", False), patch.object(demo, "KNOCK_ONLY", True):
            settings = {"DOMINO_KNOCK_SHIFT_MM": "40", "DOMINO_KNOCK_END_FORWARD_MM": "30",
                        "DOMINO_KNOCK_YAW_DEG": "20", "DOMINO_KNOCK_RIGHT_MM": "60"}
            with patch.dict(demo.os.environ, settings), contextlib.redirect_stdout(io.StringIO()):
                await self.node.join_tiles(Mock())
        self.node.knock.assert_awaited_once_with(target)
        self.node.deliver.assert_not_awaited()
        self.node.knock_path(target, ignore=(target,))
        self.assertEqual(self.node.knock_end_forward, 30)
        self.assertAlmostEqual(math.degrees(self.node.knock_yaw), 20)
        self.assertEqual(self.node.knock_right, 60)

    async def test_full_demo_starts_with_clear_strike_when_assembly_preview_fails(self):
        target = Tile(200, 0, -math.pi / 2, (3, 2))
        neighbors = [Tile(215, 80, -math.pi / 2, (3, 3)), Tile(200, 160, -math.pi / 2, (5, 2))]
        self.node.parent = SimpleNamespace(segmenter=True, classifier=True, fallen_model=True, fallen_labels=True)
        self.node.scan_row = AsyncMock(return_value=[target, *neighbors])
        self.node.knock = AsyncMock(side_effect=lambda tile: tile)
        self.node.deliver = AsyncMock(side_effect=lambda tile, destination, **kwargs: destination)
        self.node.return_to_row = AsyncMock()
        settings = {"DOMINO_KNOCK_SHIFT_MM": "40", "DOMINO_KNOCK_YAW_DEG": "30",
                    "DOMINO_KNOCK_END_FORWARD_MM": "30", "DOMINO_KNOCK_RIGHT_MM": "60"}
        output = io.StringIO()
        with patch.object(demo, "SCAN_ONLY", False), patch.object(demo, "KNOCK_ONLY", False):
            with patch.dict(demo.os.environ, settings), contextlib.redirect_stdout(output):
                await self.node.join_tiles(Mock())
        self.assertIn("Trying the best matching pair", output.getvalue())
        self.assertEqual(self.node.knock.await_args_list[0].args[0], target)
        self.node.knock_path(target, ignore=(target,))
        self.assertEqual(self.node.deliver.await_count, 2)


if __name__ == "__main__":
    unittest.main()

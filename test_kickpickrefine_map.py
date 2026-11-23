"""Offline checks for sharing kickpickrefine results with the world map."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import kickpickrefine as script
from aim_fsm import evbase
from aim_fsm.domino_map_publisher import DominoMapPublisher, observations_from_candidates
from aim_fsm.events import DataEvent, FailureEvent, SuccessEvent
from aim_fsm.utils import PoseEstimate
from aim_fsm.worldmap import DominoObj, WorldMap
from viewer.worldmap_model import WorldMapModel


class RecordedBoxes:
    def __init__(self, boxes):
        self.xyxy = torch.tensor(boxes)

    def __len__(self):
        return len(self.xyxy)


class SharedDominoMapTests(unittest.TestCase):
    def setUp(self):
        self.robot = SimpleNamespace(
            pose=PoseEstimate(0, 0, 0, 0), holding=None,
            frame_count=10, camera_image=np.zeros((480, 640, 3), dtype=np.uint8),
            particle_filter=SimpleNamespace(state='localized', LOCALIZED='localized'),
            kine=SimpleNamespace(project_to_ground=lambda x, y: np.array(
                [[400.0 - y], [320.0 - x], [0.0], [1.0]])),
            erouter=SimpleNamespace(post=Mock()),
            aruco_detector=None,
            robot0=SimpleNamespace(status={'aivision': {'objects': {'items': []}}},
                                   has_any_barrel=lambda: False,
                                   has_sports_ball=lambda: False),
        )
        self.robot.world_map = WorldMap(self.robot)
        self.publisher = DominoMapPublisher(self.robot)
        self.robot.domino_detector = self.publisher

    def standing(self, label='4-3', x=320):
        return {'quad': np.array([[x-50, 200], [x+50, 200],
                                  [x+50, 250], [x-50, 250]], dtype=np.float32),
                'label': label, 'confidence': 0.92}

    def publish(self, candidates, frame_id=10, is_fallen=False):
        observations = observations_from_candidates(candidates, frame_id, is_fallen=is_fallen)
        self.publisher.publish(observations)

    def test_all_recognized_tiles_appear_immediately_with_original_pips(self):
        self.publish([self.standing(), self.standing('2-5', x=450)])
        objects = list(self.robot.world_map.objects.values())
        self.assertEqual(len(objects), 2)
        self.assertEqual({obj.face_label for obj in objects}, {'4-3', '2-5'})
        self.assertTrue(all(isinstance(obj, DominoObj) for obj in objects))
        self.assertEqual({obj.half_counts for obj in objects}, {(4, 3), (2, 5)})
        self.assertTrue(all(obj.is_visible and obj.observed_frame == 10 for obj in objects))
        # Exercise the actual data supplied to the Qt world-map viewer.
        entry = WorldMapModel._build_object(SimpleNamespace(
            _resolve_type=WorldMapModel._resolve_type,
            _build_domino_halves=lambda obj, length, label:
                WorldMapModel._build_domino_halves(None, obj, length, label),
        ), objects[0].id, objects[0])
        self.assertEqual(entry['type'], 'domino')
        self.assertEqual([half['count'] for half in entry['domino_halves']], [4, 3])

    def test_cached_map_updates_do_not_infer_or_reproject_after_motion(self):
        self.publish([self.standing()])
        original = self.publisher.world_map_candidates()[0]
        original_xy = (original.pose.x, original.pose.y)
        self.robot.pose = PoseEstimate(500, 100, 0, math.pi / 2)
        self.robot.frame_count = 25
        with patch.object(self.publisher, 'detect', side_effect=AssertionError('new inference')):
            self.robot.world_map.candidates = []
            self.robot.world_map.make_new_domino_objects()
        cached = self.robot.world_map.candidates[0]
        self.assertEqual((cached.pose.x, cached.pose.y), original_xy)
        self.assertEqual(cached.observed_frame, 10)
        self.assertIsNone(cached.matched)
        cached.pose.x = -1234
        self.assertEqual(self.publisher.world_map_candidates()[0].pose.x, original_xy[0])

    def test_repeated_observations_reuse_id_and_empty_reading_clears_cache(self):
        self.publish([self.standing()])
        first_id = next(iter(self.robot.world_map.objects))
        self.publish([self.standing(x=322)], frame_id=11)
        self.assertEqual(list(self.robot.world_map.objects), [first_id])
        self.assertEqual(self.robot.world_map.objects[first_id].observed_frame, 11)
        self.publish([], frame_id=12)
        self.assertEqual(self.publisher.world_map_candidates(), [])
        self.assertEqual(self.publisher.latest_observations(), [])
        self.assertFalse(self.robot.world_map.objects[first_id].is_visible)

    def test_fallen_pose_is_published_without_inventing_pip_counts(self):
        self.publish([{'rect': ((320, 250), (100, 50), 0), 'mask_area': 5000}],
                     is_fallen=True)
        obj = next(iter(self.robot.world_map.objects.values()))
        self.assertTrue(obj.is_fallen)
        self.assertIsNone(obj.face_label)
        self.assertIsNone(obj.half_counts)

    def test_regular_map_updates_keep_the_published_tile_without_duplicates(self):
        self.publish([self.standing()])
        first_id = next(iter(self.robot.world_map.objects))
        for _ in range(8):
            self.robot.frame_count += 1
            self.robot.world_map.update()
        self.assertEqual(list(self.robot.world_map.objects), [first_id])
        obj = self.robot.world_map.objects[first_id]
        self.assertTrue(obj.is_visible)
        self.assertEqual(obj.observed_frame, 10)
        self.assertEqual(obj.half_counts, (4, 3))

    def test_reloading_publisher_reuses_existing_domino_objects(self):
        self.publish([self.standing()])
        first_id = next(iter(self.robot.world_map.objects))
        self.publisher.close()
        self.publisher = DominoMapPublisher(self.robot)
        self.robot.domino_detector = self.publisher
        self.publish([self.standing()], frame_id=11)
        self.assertEqual(list(self.robot.world_map.objects), [first_id])

    def test_existing_unknown_pips_can_be_replaced_by_a_real_reading(self):
        self.publish([self.standing()])
        obj = next(iter(self.robot.world_map.objects.values()))
        obj.face_label = '?'
        obj.half_counts = None
        self.publish([self.standing()], frame_id=11)
        self.assertEqual(len(self.robot.world_map.objects), 1)
        self.assertEqual(self.robot.world_map.objects[obj.id].half_counts, (4, 3))

    def test_fallen_reading_updates_nearby_standing_object_without_guessing_label(self):
        self.publish([self.standing()])
        first_id = next(iter(self.robot.world_map.objects))
        self.publish([{'rect': ((320, 250), (100, 50), 0)}], frame_id=11, is_fallen=True)
        self.assertEqual(list(self.robot.world_map.objects), [first_id])
        obj = self.robot.world_map.objects[first_id]
        self.assertTrue(obj.is_fallen)
        self.assertIsNone(obj.half_counts)

    def test_normal_detector_path_still_projects_observations(self):
        observation = observations_from_candidates([self.standing()], 10)[0]
        self.robot.domino_detector = SimpleNamespace(detect=Mock(return_value=[observation]))
        self.robot.world_map.candidates = []
        self.robot.world_map.make_new_domino_objects()
        self.robot.domino_detector.detect.assert_called_once()
        self.assertEqual(self.robot.world_map.candidates[0].half_counts, (4, 3))

    def test_same_image_position_is_transformed_using_observation_robot_pose(self):
        self.robot.pose = PoseEstimate(100, 200, 0, math.pi / 2)
        self.publish([self.standing()])
        obj = next(iter(self.robot.world_map.objects.values()))
        self.assertAlmostEqual(obj.pose.x, 100)
        self.assertAlmostEqual(obj.pose.y, 200 + 1.55 * 150 - 58.4)

    def program_without_starting_robot(self):
        program = object.__new__(script.kickpickrefine)
        program._robot = self.robot
        program.children = {}
        program.start_node = None
        program.name = 'offline-test'
        program.map_publisher = self.publisher
        program.target_data = None
        program.clahe = None
        return program

    def run_tracking(self, labels):
        program = self.program_without_starting_robot()
        boxes = [[200 + index * 150, 200, 300 + index * 150, 250]
                 for index in range(len(labels))]
        program.segmenter = Mock(return_value=[SimpleNamespace(
            obb=None, boxes=RecordedBoxes(boxes))])
        predictions = iter((int(n), 0.95) for label in labels for n in label.split('-'))
        program.predict_half = lambda tensor: next(predictions)
        # Node.start posts events into a recording router; no FSM transitions run.
        with patch.object(evbase, 'robot_for_loading', self.robot, create=True), \
                patch.object(script, 'imshow'):
            node = program.TrackAndIdentify().set_parent(program)
            node.start()
        return program

    def test_tracking_shares_results_but_only_selects_three_four(self):
        program = self.run_tracking(['4-3', '2-5'])
        program.segmenter.assert_called_once()
        self.assertEqual(program.target_data['label'], '4-3')
        self.assertIsInstance(self.robot.erouter.post.call_args.args[0], DataEvent)
        self.assertEqual({obj.face_label for obj in self.robot.world_map.objects.values()},
                         {'4-3', '2-5'})

    def test_other_pips_are_displayed_without_becoming_motion_targets(self):
        program = self.run_tracking(['2-5'])
        self.assertIsNone(program.target_data)
        self.assertIsInstance(self.robot.erouter.post.call_args.args[0], FailureEvent)
        self.assertEqual(next(iter(self.robot.world_map.objects.values())).half_counts, (2, 5))

    def test_map_publication_failure_does_not_change_motion_target_selection(self):
        with patch.object(self.publisher, 'publish', side_effect=RuntimeError('map unavailable')):
            program = self.run_tracking(['4-3'])
        self.assertEqual(program.target_data['label'], '4-3')
        self.assertIsInstance(self.robot.erouter.post.call_args.args[0], DataEvent)

    def test_fallen_node_shares_its_existing_inference(self):
        program = self.program_without_starting_robot()
        program.actual_retreated_mm = 60.0
        mask = torch.zeros((1, 160, 160))
        mask[:, 70:85, 65:95] = 1
        program.fallen_model = SimpleNamespace(predict=Mock(return_value=[
            SimpleNamespace(masks=SimpleNamespace(data=mask))]))
        with patch.object(evbase, 'robot_for_loading', self.robot, create=True), \
                patch.object(script.time, 'sleep'), patch.object(script, 'imshow'):
            node = program.DetectTarget().set_parent(program)
            node.start()
        program.fallen_model.predict.assert_called_once()
        self.assertIsInstance(self.robot.erouter.post.call_args.args[0], SuccessEvent)
        self.assertTrue(next(iter(self.robot.world_map.objects.values())).is_fallen)
        self.assertGreater(program.measured_forward_cm, 0)

    def test_constructor_disables_independent_map_detector(self):
        def base_init(program, **kwargs):
            self.assertEqual(kwargs, {'domino': False})
            program._robot = self.robot
            program.domino = False
        with patch.object(script.StateMachineProgram, '__init__', base_init), \
                patch.object(script, 'YOLO') as model, \
                patch.object(script.kickpickrefine, '_load_classifier'):
            program = script.kickpickrefine()
            program._init_domino_detector()
        self.assertIs(self.robot.domino_detector, program.map_publisher)
        self.assertEqual([call.args[0] for call in model.call_args_list],
                         [script.SEG_MODEL_PATH, script.FALLEN_MODEL_PRIMARY])


if __name__ == '__main__':
    unittest.main()

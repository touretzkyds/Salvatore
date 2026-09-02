import math
import copy
import numpy as np
import time
import datetime
import threading
import cv2

from .geometry import *
from .utils import *
from .camera import AIVISION_RESOLUTION_SCALE

# Dominoes are long and thin, so data association has to weigh orientation much
# more heavily than it does for barrels or balls.
DOMINO_ASSOCIATION_ANGLE_THRESHOLD = 35 * pi/180
DOMINO_ASSOCIATION_ANGLE_WEIGHT_MM = 40.0
DOMINO_MAX_ASSOCIATION_COST = 900.0
DOMINO_PENDING_COST_THRESHOLD = 225.0

# Empirical correction for projecting an image point onto the ground plane.
GROUND_PROJECTION_K1 = 1.55
GROUND_PROJECTION_K2 = -58.4

# Domino calibration
KNOWN_LENGTH = 4.8      # Domino length in cm
KNOWN_LENGTH_MM = 48.0  # Domino length in mm
FOCAL_LENGTH = 396.3    # Calibrated focal length in pixels


def normalize_axis_angle(theta):
    "Normalize theta to [-pi, pi) to preserve full 360-degree orientations."
    if theta is None:
        return None
    return (theta + pi) % (2 * pi) - pi


def axis_angle_distance(theta_a, theta_b):
    "Angular distance treating theta and theta+pi as the same axis."
    if theta_a is None or theta_b is None:
        return 0.0
    return abs(wrap_angle(2 * (theta_a - theta_b))) / 2


def align_axis_angle(theta, reference):
    "Return theta or theta+pi, whichever is closer to reference."
    if theta is None:
        return None
    theta0 = normalize_axis_angle(theta)
    if reference is None:
        return theta0
    theta1 = normalize_axis_angle(theta0 + pi)
    if abs(wrap_angle(theta0 - reference)) <= abs(wrap_angle(theta1 - reference)):
        return theta0
    return theta1


HALF_ORDERED_ATTRS = ('half_counts', 'half_image_centers', 'half_world_centers')


def reverse_half_ordering(attr, value):
    """Reverse a domino attribute that is ordered along the tile's axis.

    Needed when an observation's heading is folded 180 degrees onto the tracked
    heading: the halves must be swapped too, or the pip counts end up describing
    the opposite ends from the ones pose.theta points at.
    """
    if value is None:
        return None
    if attr in HALF_ORDERED_ATTRS:
        return tuple(reversed(tuple(value)))
    if attr == 'face_label' and isinstance(value, str) and value.count('-') == 1:
        left, right = value.split('-')
        return f'{right}-{left}'
    return value


def _mean_xy(points):
    if len(points) == 0:
        return (0.0, 0.0)
    sx = sum(pt[0] for pt in points)
    sy = sum(pt[1] for pt in points)
    return (sx / len(points), sy / len(points))


class WorldObject():
    def __init__(self, id=None, name=None, x=0, y=0, z=0, theta=None, is_visible=False, is_fixed=False):
        self.id = id
        self.pose = PoseEstimate(x, y, z, theta)
        self.name = name or self.__class__.__name__
        self.matched = None  # matching object from data association
        self.is_fixed = is_fixed   # True for walls and markers in predefined maps
        self.is_obstacle = True # for path planning
        self.is_visible = is_visible
        self.is_missing = False # expect to see it but we don't
        self.is_valid = True
        self.held_by = None
        self.is_foreign = False # for shared maps; unused for now
        if is_visible:
            self.pose_confidence = +1
        else:
            self.pose_confidence = -1

    def __repr__(self):
        vis = 'visible' if self.is_visible else 'missing' if self.is_missing else 'unseen'
        held = " held" if self.held_by else ""
        return f'<{self.id or self.name} {vis} at ({self.pose.x:.1f}, {self.pose.y:.1f}){held}>'

    def update_matched_object(self,robot):
        "Update the matched world_map object with info from this candidate."
        self.matched.is_visible = True
        if self.matched.is_fixed or robot.particle_filter.state != robot.particle_filter.LOCALIZED:
            return
        MIN_MEASUREMENT_NOISE = 5
        measurement_noise = max(MIN_MEASUREMENT_NOISE, math.sqrt(self.sensor_distance)/2)
        halves_reversed = False
        if self.matched is not robot.holding:  # pose update will be done by update_held_object()
            pose_update = self.pose
            if isinstance(self, DominoObj):
                # A domino seen from the other side reports a heading 180 degrees
                # away; fold it onto the tracked heading before filtering.
                aligned_theta = align_axis_angle(self.pose.theta,
                                                 getattr(self.matched.pose, 'theta', None))
                halves_reversed = (aligned_theta is not None and self.pose.theta is not None
                                   and abs(wrap_angle(aligned_theta - self.pose.theta)) > pi/2)
                pose_update = Pose(self.pose.x, self.pose.y, self.pose.z, aligned_theta)
            if not hasattr(self.matched.pose, 'update'):
                self.matched.pose = PoseEstimate(self.matched.pose)
            self.matched.pose.update(pose_update, measurement_noise)
            if isinstance(self, DominoObj):
                normalized_theta = normalize_axis_angle(self.matched.pose.theta)
                self.matched.pose.theta = normalized_theta
                if hasattr(self.matched.pose, 'kf_theta'):
                    self.matched.pose.kf_theta.state = normalized_theta
        if hasattr(self, 'spec'):
            self.matched.spec = self.spec
        if hasattr(self, 'marker'):
            self.matched.marker = self.marker
        if hasattr(self, 'seen_markers'):
            self.matched.seen_markers = self.seen_markers
        if hasattr(self, 'sensor_distance'):
            self.matched.sensor_distance = self.sensor_distance
        if hasattr(self, 'sensor_bearing'):
            self.matched.sensor_bearing = self.sensor_bearing
        if hasattr(self, 'sensor_orient'):
            self.matched.sensor_orient = self.sensor_orient
        if hasattr(self, 'wall'):
            self.matched.wall = self.wall.matched
        for attr in ('image_center', 'image_quad', 'image_axis', 'image_divider',
                     'half_counts', 'half_image_centers', 'half_world_centers',
                     'mask_area', 'face_label', 'face_confidence', 'confidence',
                     'is_fallen', 'observed_frame'):
            if hasattr(self, attr):
                value = getattr(self, attr)
                if halves_reversed:
                    value = reverse_half_ordering(attr, value)
                setattr(self.matched, attr, value)

class BarrelObj(WorldObject):
    def __init__(self, spec=None, id=None, x=0, y=0):
        if id is None and spec and 'id' in spec and isinstance(spec['id'], str):
            id = spec['id']
        super().__init__(id=id, x=x, y=y)
        self.spec = spec
        self.name = spec['name']
        self.diameter = 22 # mm
        self.height = 25 # mm

class OrangeBarrelObj(BarrelObj):
    pass

class BlueBarrelObj(BarrelObj):
    pass

class SportsBallObj(WorldObject):
    def __init__(self, spec=None, id=None, x=0, y=0):
        if id is None and spec and 'id' in spec and isinstance(spec['id'], str):
            id = spec['id']
        super().__init__(id=id, x=x, y=y)
        self.spec = spec
        self.name = spec['name']
        self.diameter = 25.0 # mm
        self.z = self.diameter / 2

class RobotObj(WorldObject):
    def __init__(self, spec=None, id=None, x=0, y=0, theta=0):
        super().__init__(id=id, x=x, y=y, theta=theta)
        self.spec = spec
        self.name = spec['name']

class AprilTagObj(WorldObject):
    def __init__(self, spec=None, id=None, x=0, y=0, theta=0):
        if id is None and spec and 'id' in spec:
            id = spec['id']
        super().__init__(id=id, x=x, y=y, theta=theta)
        self.spec = spec
        self.name = spec['name']
        self.tag_id = spec['id']
        self.base_diameter = 22 # mm
        self.width = 38 # mm

    def __repr__(self):
        vis = 'visible' if self.is_visible else 'missing' if self.is_missing else 'unseen'
        return f'<{self.id or self.name} {vis} at ({self.pose.x:.1f}, {self.pose.y:.1f}) @ {self.pose.theta*180/pi:.1f} deg.>'
    

class AprilTag0Obj(AprilTagObj):
    pass

class AprilTag1Obj(AprilTagObj):
    pass

class AprilTag2Obj(AprilTagObj):
    pass

class AprilTag3Obj(AprilTagObj):
    pass

class AprilTag4Obj(AprilTagObj):
    pass

class ArucoMarkerObj(WorldObject):
    def __init__(self, spec, x=0, y=0, z=0, theta=0, **kwargs):
        super().__init__(x=x, y=y, z=z, theta=theta, **kwargs)
        self.name = spec['name']
        self.marker = spec['marker']
        self.marker_id = spec['id']
        self.marker_string = 'ArucoMarker-' + str(spec['id'])
        self.pose_confidence = +1

    def __repr__(self):
        if self.pose_confidence >= 0:
            vis = 'visible' if self.is_visible else 'missing' if self.is_missing else 'unseen'
            fix = ' fixed' if self.is_fixed else ''
            return '<ArucoMarkerObj %s: (%.1f, %.1f, %.1f) @ %d deg.%s %s>' % \
                (self.marker_id, self.pose.x, self.pose.y, self.pose.z, self.pose.theta*180/pi, fix, vis)
        else:
            return f'<ArucoMarkerObj {self.marker_id}: position unknown>'
        

class WallObj(WorldObject):

    def __init__(self, wall_spec, x=0, y=0, z=0, theta=0):
        super().__init__(x=x, y=y, z=z, theta=theta)
        self.wall_spec = wall_spec
        self.name = wall_spec.label
        self.length = wall_spec.length
        self.height = wall_spec.height
        self.is_fixed = False

    def __repr__(self):
        vis = 'visible' if self.is_visible else 'unseen'
        return f'<WallObj {self.name} ({self.pose.x:.1f}, {self.pose.y:.1f}) @ {self.pose.theta*180/pi:.1f} deg. {vis}>'

    ALIGNMENT_THRESHOLD = 25 * pi/180 # 25 degrees: aruco markers can only differ by this much

    def is_wall_aligned(self, obj):
        """An aruco marker or candidate wall is wall-aligned if the
        sensor_orient values match, but could be off by pi if marker
        is on the back of the wall."""
        result = abs(wrap_angle(self.sensor_orient - obj.sensor_orient)) < self.ALIGNMENT_THRESHOLD or \
            (isinstance(obj,ArucoMarkerObj) and \
             abs(wrap_angle(self.sensor_orient + pi - obj.sensor_orient)) < self.ALIGNMENT_THRESHOLD)
        if result is False: pass
            # print(f'is_wall_aligned {self.name} {neaten(self.sensor_orient*180/pi)} with {obj.name}' +
            #       f' {neaten(obj.sensor_orient*180/pi)}' +
            #       f' diff = {neaten(abs(wrap_angle(self.sensor_orient - obj.sensor_orient))*180/pi)}  result: {result}')
        return result


class WallSpec():
    def __init__(self, wall_marker_dict, label=None, length=100, height=210, marker_specs=dict(), doorways=dict()):
        self.length = length
        self.height = height
        self.marker_specs = marker_specs
        self.doorways = doorways
        marker_id_numbers = list(marker_specs.keys())
        self.label = label or f'Wall-{min(marker_id_numbers)}'
        for id in marker_id_numbers:
            wall_marker_dict[id] = self
        wall_marker_dict[self.label] = self


class DoorwayObj(WorldObject):
    def __init__(self, wall, index):
        name = f'Doorway-{wall.name[5:]}:{index}'
        super().__init__(name=name, is_visible=wall.is_visible)
        door_spec = wall.wall_spec.doorways[index]
        self.door_width = door_spec['width']
        self.wall = wall
        self.index = index  # which doorway is this?  0, 1, ...
        self.is_obstacle = False
        self.update()

    def update(self):
        door_spec = self.wall.wall_spec.doorways[self.index]
        self.pose = copy.deepcopy(self.wall.pose)
        self.sensor_distance = self.wall.sensor_distance

    def __repr__(self):
        vis = 'visible' if self.is_visible else 'unseen'
        if self.pose_confidence >= 0:
            return '<DoorwayObj %s: (%.1f,%.1f) @ %.1f deg. %s>' % \
                (self.id, self.pose.x, self.pose.y, self.pose.theta*180/pi, vis)
        else:
            return '<DoorwayObj %s: position unknown>' % self.id

class RoomObj(WorldObject):
    def __init__(self, name,
                 points=np.resize(np.array([0,0,0,1]),(4,4)).transpose(),
                 floor=1, door_ids=[], connections=[]):
        "points should be four points in homogeneous coordinates forming a convex polygon"
        id = 'Room-' + name
        self.name = name
        x,y,z,s = points.mean(1)
        super().__init__(id=id, x=x, y=y)
        self.points = points
        self.floor = floor
        self.door_ids = door_ids
        self.connections = connections
        self.is_obstacle = False
        self.is_fixed = True

    def __repr__(self):
        return '<RoomObj %s: (%.1f,%.1f) floor=%s>' % (self.id, self.pose.x, self.pose.y, self.floor)

    def get_bounding_box(self):
        mins = self.points.min(1)
        maxs = self.points.max(1)
        return ((mins[0],mins[1]), (maxs[0],maxs[1]))


class DominoObj(WorldObject):
    """A double-six domino tile seen by robot.domino_detector.

    face_label is the "high-low" pip reading, e.g. '6-3', and is None until the
    detector runs with a label provider (domino_labeling=True).
    """
    def __init__(self, id=None, x=0, y=0, z=0, theta=0,
                 face_label=None, face_confidence=None, is_fallen=False):
        super().__init__(id=id, name='Domino', x=x, y=y, z=z,
                         theta=normalize_axis_angle(theta))
        self.length = KNOWN_LENGTH_MM
        self.is_fallen = is_fallen
        self.width = 24.0
        if is_fallen:
            self.height = 8.0
        else:
            self.height = 24.0
        self.thickness = 8.0
        self.is_obstacle = True
        self.face_label = face_label
        self.face_confidence = face_confidence
        self.confidence = None
        self.image_center = None
        self.image_quad = None
        self.image_axis = None
        self.image_divider = None
        self.half_counts = None
        self.half_image_centers = None
        self.half_world_centers = None
        self.mask_area = None
        # Camera frame this tile's pips were last read from. Consumers use it
        # to tell a fresh reading from the same reading looked at twice.
        self.observed_frame = None

    @property
    def pips(self):
        "The two half-face pip counts as a tuple, or None if not yet read."
        if not self.half_counts:
            return None
        if any(count is None for count in self.half_counts):
            return None
        return tuple(int(count) for count in self.half_counts)

    def __repr__(self):
        vis = 'visible' if self.is_visible else 'missing' if self.is_missing else 'unseen'
        theta_deg = 0.0 if self.pose.theta is None else self.pose.theta * 180 / pi
        face = f' {self.face_label}' if self.face_label else ''
        status = ' fallen' if self.is_fallen else ' standing'
        return f'<{self.id or self.name}{face}{status} {vis} at ({self.pose.x:.1f}, {self.pose.y:.1f}) @ {theta_deg:.1f} deg.>'


################################################################

class WorldMap():

    def __init__(self,robot):
        self.robot = robot
        self._lock = threading.RLock()
        self.objects = dict()
        self.pending_objects = dict()
        self.missing_objects = []
        self.shared_objects = dict()
        self.name_counts = dict()  # For generating new object names
        self.last_held_time = -1
        self.visibility_paused = False

    def __repr__(self):
        with self._lock:
            count = len(self.objects)
        return f'<WorldMap with {count} objects>'

    def snapshot_objects(self):
        with self._lock:
            snapshot = {}
            for key, obj in self.objects.items():
                try:
                    cloned = copy.copy(obj)
                    pose = getattr(obj, "pose", None)
                    if pose is not None:
                        try:
                            cloned.pose = PoseEstimate(pose)
                        except Exception:
                            cloned.pose = copy.copy(pose)
                    snapshot[key] = cloned
                except Exception:
                    snapshot[key] = obj
            return snapshot

    def clear(self):
        with self._lock:
            self.robot.particle_filter.clear_landmarks()
            self.objects.clear()
            self.pending_objects.clear()
            self.missing_objects = []
            self.shared_objects.clear()
            self.name_counts.clear()
        
    def pause_visibility(self, value=True):
        """Turn off visibility of objects when the robot is moving.  We won't
        turn it back on until the robot has stopped AND we have processed a new
        camera frame so visibilities are updated."""
        with self._lock:
            if self.visibility_paused != value:
                self.visibility_paused = value

    def update(self):
        with self._lock:
            if self.visibility_paused:
                return
            self.updated_objects = []
            self.make_new_objects_from_vision()
            self.associate_objects()
            self.update_associated_objects()
            self.detect_missing_objects()
            self.process_unassociated_objects()
            self.update_visibilities()
            self.update_holding()

    def make_new_objects_from_vision(self):
        self.candidates = list()
        self.make_new_aiv_objects()
        if self.robot.aruco_detector:
            self.make_new_wall_objects()
            self.make_new_aruco_objects()
        if getattr(self.robot, 'domino_detector', None):
            self.make_new_domino_objects()

    def make_vision_object(self, spec):
        if spec['name'] == 'OrangeBarrel':
            obj = OrangeBarrelObj(spec)
        elif spec['name'] == 'BlueBarrel':
            obj = BlueBarrelObj(spec)
        elif spec['name'] == 'SportsBall':
            obj = SportsBallObj(spec)
        elif spec['name'] == 'Robot':
            obj = RobotObj(spec)
        elif spec['name'].startswith('AprilTag'):
            obj = AprilTagObj(spec)
        else:
            print(f"ERROR **** spec = {spec}")
            return None
        return obj

    def make_new_aiv_objects(self):
        objspecs = self.robot.robot0.status['aivision']['objects']['items']
        for spec in objspecs:
            if spec['type_str'] == 'aiobj':
                base_name = spec['name']
            elif spec['type_str'] == 'tag':
                if 0 <= spec['id'] <= 4:
                    base_name = 'AprilTag-' + repr(spec['id'])
                    spec['name'] = base_name
                else:
                    #print('*** BAD TAG:', spec)
                    continue
            else:
                print(f'*** Unknown: spec={spec}')
                continue
            if spec['name'] == 'Robot':   # avoid spurious robot creation for now
                pass  # continue
            obj = self.make_vision_object(spec)
            obj.is_visible = True
            # Calculate midpoint of bottom edge, which we assume is on the floor
            cx = (spec['originx'] + spec['width']/2) * AIVISION_RESOLUTION_SCALE
            # correct height for possible occlusion by foreground object
            corr_height = max(spec['height'], spec['width']*1.10)
            cy = (spec['originy'] + corr_height) * AIVISION_RESOLUTION_SCALE
            if isinstance(obj, AprilTagObj):
                cy += spec['height'] * 2 * AIVISION_RESOLUTION_SCALE
            hit = self.robot.kine.project_to_ground(cx, cy)
            # Correct for distortion: constants calculated from measurements with 24.3 degree camera tilt
            K1 = 1.55; K2 = -58.4
            oldhit = hit.copy()
            #hit[0] = K1 * hit[0] + K2
            adjhit = hit.copy()
            # offset hit by half the object thickness
            angle = atan2(hit[1,0], hit[0,0])
            if obj.__dict__.get('diameter'):
                half_diameter = obj.diameter / 2
                increment = point(cos(angle) * half_diameter, sin(angle) * half_diameter, 0)
                #print(f'{hit=}  {increment=}  {hit+increment=}')
                hit += increment
            #print(f'{oldhit[0,0]=}  {adjhit[0,0]=}  {hit[0,0]=}')
            # convert to world coordinates
            robotpos = point(self.robot.pose.x, self.robot.pose.y)
            objpos = aboutZ(self.robot.pose.theta).dot(hit) + robotpos
            x = objpos[0][0]
            y = objpos[1][0]
            distance = ((x - self.robot.pose.x)**2 + (y - self.robot.pose.y)**2) ** 0.5
            MAX_DISTANCE = 300 # anything further than this is a spurious detection
            if distance > MAX_DISTANCE:
                continue
            obj.sensor_distance = distance
            if isinstance(obj, AprilTagObj):
                tag_angle_correction_factor = 4  # guesstimate
                angle = spec['angle'] - (0 if spec['angle'] < 180 else 360)
                theta = self.robot.pose.theta - angle / 180 * pi * tag_angle_correction_factor
            else:
                theta = None
            obj.pose = Pose(x, y, 0, theta)
            if self.check_spec_indicates_held(obj):
                self.reposition_held_object(obj)
            self.candidates.append(obj)

    def project_image_point_to_world(self, cx, cy):
        """Project an image pixel onto the ground plane.

        Returns (hit, objpos): hit is robot-relative, objpos is world coordinates.
        Applies the distortion correction calibrated for the 24.3 degree camera tilt.
        """
        hit = self.robot.kine.project_to_ground(cx, cy)
        hit[0] = GROUND_PROJECTION_K1 * hit[0] + GROUND_PROJECTION_K2
        robotpos = point(self.robot.pose.x, self.robot.pose.y)
        objpos = aboutZ(self.robot.pose.theta).dot(hit) + robotpos
        return hit, objpos

    def make_new_domino_objects(self):
        "Turn this frame's domino detections into DominoObj candidates."
        detector = getattr(self.robot, 'domino_detector', None)
        image = getattr(self.robot, 'camera_image', None)
        if detector is None or image is None:
            return
        try:
            observations = detector.detect(image, frame_id=getattr(self.robot, 'frame_count', None))
        except Exception as exc:
            print(f'*** Domino detector failed: {exc}')
            return

        for obs in observations:
            quad = getattr(obs, 'quad', None)
            quad = np.array(quad, dtype=np.float32) if quad is not None else None
            if quad is None or len(quad) != 4:
                cnt = getattr(obs, 'contour', None)
                if cnt is not None and len(cnt) >= 4:
                    quad = cv2.boxPoints(cv2.minAreaRect(cnt))
                else:
                    cx, cy = obs.center_xy
                    quad = np.array([[cx-20, cy-10], [cx+20, cy-10],
                                     [cx+20, cy+10], [cx-20, cy+10]], dtype=np.float32)

            # The lowest edge of the tile in the image is the one resting on the table.
            cx, _ = obs.center_xy
            bottom_cy = float(np.max(quad[:, 1]))
            hit, objpos = self.project_image_point_to_world(cx, bottom_cy)
            world_x = float(objpos[0][0])
            world_y = float(objpos[1][0])
            local_x = float(hit[0][0])
            local_y = float(hit[1][0])

            calibrated_distance_mm = math.hypot(local_x, local_y)
            MAX_DOMINO_DISTANCE = 800.0
            if calibrated_distance_mm > MAX_DOMINO_DISTANCE:
                continue

            axis_endpoints = getattr(obs, 'axis_endpoints', None)
            if axis_endpoints is not None:
                (pt0_x, pt0_y), (pt1_x, pt1_y) = axis_endpoints
            else:
                # Fall back to the bottom edge of the quad, left endpoint first.
                sorted_by_y = sorted(quad, key=lambda pt: pt[1], reverse=True)
                p_base1, p_base2 = sorted_by_y[0], sorted_by_y[1]
                if p_base1[0] > p_base2[0]:
                    p_base1, p_base2 = p_base2, p_base1
                (pt0_x, pt0_y), (pt1_x, pt1_y) = p_base1, p_base2
            hit0, _ = self.project_image_point_to_world(pt0_x, pt0_y)
            hit1, _ = self.project_image_point_to_world(pt1_x, pt1_y)
            local_theta = math.atan2(float(hit1[1][0] - hit0[1][0]),
                                     float(hit1[0][0] - hit0[0][0]))
            world_theta = normalize_axis_angle(self.robot.pose.theta + local_theta)

            is_fallen = bool(getattr(obs, 'is_fallen', False))
            z_pos = 4.0 if is_fallen else 12.0

            obj = DominoObj(x=world_x, y=world_y, z=z_pos, theta=world_theta,
                            face_label=obs.face_label,
                            face_confidence=obs.face_confidence,
                            is_fallen=is_fallen)
            if not hasattr(obj.pose, 'update'):
                obj.pose = PoseEstimate(obj.pose)
            obj.half_counts = getattr(obs, 'half_counts', None)
            obj.half_image_centers = getattr(obs, 'half_image_centers', None)
            obj.sensor_distance = calibrated_distance_mm
            obj.sensor_bearing = math.atan2(local_y, local_x)
            obj.sensor_orient = world_theta
            obj.image_center = tuple(float(v) for v in obs.center_xy)
            obj.image_quad = tuple((float(px), float(py)) for (px, py) in quad)
            if axis_endpoints is not None:
                obj.image_axis = ((float(axis_endpoints[0][0]), float(axis_endpoints[0][1])),
                                  (float(axis_endpoints[1][0]), float(axis_endpoints[1][1])))
            divider = getattr(obs, 'divider_endpoints', None)
            if divider is not None:
                obj.image_divider = ((float(divider[0][0]), float(divider[0][1])),
                                     (float(divider[1][0]), float(divider[1][1])))
            obj.mask_area = float(getattr(obs, 'mask_area', 0.0))
            obj.confidence = float(getattr(obs, 'confidence', 1.0))
            obj.observed_frame = getattr(self.robot, 'frame_count', None)
            obj.is_visible = True
            self.candidates.append(obj)

    def make_new_aruco_objects(self):
        camera_offset_vector = np.array([0, 0, self.robot.kine.camera_from_origin])
        detector = self.robot.aruco_detector
        if hasattr(detector, "snapshot_seen_markers"):
            seen_markers = detector.snapshot_seen_markers()
        else:
            seen_markers = detector.seen_marker_objects.copy()
        for (id,marker) in seen_markers.items():
            name = f'ArucoMarker-{id}'
            spec = {'name': name, 'id': id, 'marker': marker}
            sensor_coords = marker.camera_coords + camera_offset_vector
            sensor_distance = math.sqrt(sensor_coords[0]**2 + sensor_coords[2]**2)
            sensor_bearing = atan2(sensor_coords[0], sensor_coords[2])
            sensor_orient = wrap_angle(pi - marker.euler_angles[1])
            theta = self.robot.pose.theta
            obj = ArucoMarkerObj(spec)
            obj.pose = Pose(self.robot.pose.x + sensor_distance * cos(theta + sensor_bearing),
                            self.robot.pose.y + sensor_distance * sin(theta + sensor_bearing),
                            marker.aruco_parent.marker_size / 2,  # *** TEMPORARY HACK ***
                            wrap_angle(self.robot.pose.theta + sensor_orient))
            obj.sensor_distance = sensor_distance
            obj.sensor_bearing = sensor_bearing
            obj.sensor_orient = sensor_orient
            obj.is_visible = True
            self.candidates.append(obj)

    def make_new_wall_objects(self):
        detector = self.robot.aruco_detector
        if hasattr(detector, "snapshot_seen_markers"):
            seen = detector.snapshot_seen_markers()
        else:
            seen = detector.seen_marker_objects.copy()
        wall_markers = dict()
        for (id,marker) in seen.items():
            if id in self.robot.world_map.wall_marker_dict:
                spec = self.robot.world_map.wall_marker_dict[id]
                if spec.label not in wall_markers:
                    wall_markers[spec.label] = list()
                wall_markers[spec.label].append((id,marker))
        for (wall_id, markers) in wall_markers.items():
            orients = [marker[1].euler_angles[1] for marker in markers]
            orig_orients = copy.copy(orients)
            if len(orients) == 1:
                # one marker is enough to update a wall if we're localized
                if self.robot.particle_filter.state != self.robot.particle_filter.LOCALIZED:
                    continue
            elif len(orients) == 2:
                if abs(wrap_angle(orients[0] - orients[1])) > WallObj.ALIGNMENT_THRESHOLD:
                    # with two markers that disagree, we can't tell which is the outlier, so punt
                    #print(f'wall {wall_id} marker outlier: {[o*180/pi for o in orients]}')
                    continue
            else:
                orients_consistent = False
                while not orients_consistent:
                    n = len(orients)
                    orients_consistent = True
                    for i in range(n):
                        exceeds = [abs(wrap_angle(orients[i] - orients[(i+j+1)%n])) > WallObj.ALIGNMENT_THRESHOLD
                                   for j in range(n-1)]
                        if all(exceeds):
                            #print('marker',i,' outlier:', orients)
                            del orients[i]
                            del markers[i]
                            orients_consistent = False
                            break
                if len(orients) < 2:
                    print('outlier removal left us one marker:', markers)
            wall = self.infer_wall_from_corners_lists(wall_id, markers)
            if wall is None:
                continue
            wall.aruco_orients = orients
            wall.seen_markers = markers
            self.candidates.append(wall)
            #print('candidate:', wall, 'orients(deg)=', [o*180/pi for o in orients])
            # Don't make doorways until wall is confirmed in world map
            if [k for k in self.objects.keys() if k.startswith(wall.name)]:
                self.make_doorways_from_wall(wall)

    def infer_wall_from_corners_lists(self, wall_id, markers):
        wall_spec = self.robot.world_map.wall_marker_dict[wall_id]
        marker_size = self.robot.aruco_detector.marker_size
        world_points = []
        image_points = []
        last_solution = None
        for (id, marker) in markers:
            length = wall_spec.length
            side = wall_spec.marker_specs[id]['side']
            cx = wall_spec.marker_specs[id]['x']
            cy = wall_spec.marker_specs[id]['y']
            world_points.append((cx-marker_size/2 - length/2, cy+marker_size/2, 0.))
            world_points.append((cx+marker_size/2 - length/2, cy+marker_size/2, 0.))
            world_points.append((cx+marker_size/2 - length/2, cy-marker_size/2, 0.))
            world_points.append((cx-marker_size/2 - length/2, cy-marker_size/2, 0.))

            corners = marker.corners[0]
            image_points.append(corners[0])
            image_points.append(corners[1])
            image_points.append(corners[2])
            image_points.append(corners[3])

            # Find rotation and translation vector from camera frame using SolvePnP
            try:
                (success, rvec, tvec) = cv2.solvePnP(np.array(world_points, dtype=np.float64),
                                                     np.array(image_points, dtype=np.float64),
                                                     self.robot.camera.camera_matrix,
                                                     self.robot.camera.distortion_array)
            except Exception as e:
                print('*** SolvePnP exception', e, '\n',
                      'world_points=', world_points, '\n',
                      'image_points=', image_points)
                continue
            if success:
                last_solution = (rvec, tvec, side)
        if last_solution is None:
            return None
        rvec, tvec, side = last_solution
        rotationm, jacob = cv2.Rodrigues(rvec)
        euler_angles = rotation_matrix_to_euler_angles(rotationm)
        wall_orient = euler_angles[1]
        tvec[2][0] += self.robot.kine.camera_from_origin  # want distance from base frame not camera

        sensor_coords = (-tvec[0,0], -tvec[1,0], tvec[2,0])
        sensor_distance = math.sqrt(sensor_coords[0]**2 + sensor_coords[2]**2)
        sensor_bearing = atan2(sensor_coords[0], sensor_coords[2])
        # Flip wall orientation to match ArUcos for worldmap
        sensor_orient = wrap_angle(pi - wall_orient) if side > 0 else -wall_orient
        theta = self.robot.pose.theta
        wall = WallObj(wall_spec)
        wall.pose = Pose(self.robot.pose.x + sensor_distance * cos(theta + sensor_bearing),
                         self.robot.pose.y + sensor_distance * sin(theta + sensor_bearing),
                         0,
                         wrap_angle(self.robot.pose.theta + sensor_orient))
        wall.sensor_distance = sensor_distance
        wall.sensor_bearing = sensor_bearing
        wall.sensor_orient = sensor_orient
        wall.is_visible = True
        return wall

    def make_doorways_from_wall(self, wall):
        for (index, door_spec) in wall.wall_spec.doorways.items():
            door = DoorwayObj(wall, index)
            self.candidates.append(door)

    def generate_doorway_list(self):
        "Used by path-planner.py"
        doorways = []
        for (key,obj) in self.objects.items():
            if isinstance(obj,DoorwayObj):
                w = obj.door_width / 2
                doorway_threshold_theta = obj.pose.theta + pi/2
                dx = w * cos(doorway_threshold_theta)
                dy = w * sin(doorway_threshold_theta)
                ox = obj.pose.x
                oy = obj.pose.y
                doorways.append((obj, ((ox-dx, oy-dy), (ox+dx, oy+dy))))
        return doorways

    def associate_objects(self):
        obj_types = list(set(type(obj) for obj in self.candidates))
        for otype in obj_types:
            self.associate_objects_of_type(otype)

    def association_cost(self, new_obj, old_obj):
        if isinstance(new_obj, DominoObj):
            # Dominoes sit close together on the table, so orientation is the
            # main cue that tells two neighboring tiles apart.
            angle_cost = axis_angle_distance(new_obj.pose.theta, old_obj.pose.theta)
            if angle_cost > DOMINO_ASSOCIATION_ANGLE_THRESHOLD:
                return np.inf
            dist_sq = ((new_obj.pose.x - old_obj.pose.x)**2 + (new_obj.pose.y - old_obj.pose.y)**2)
            return dist_sq + (DOMINO_ASSOCIATION_ANGLE_WEIGHT_MM * angle_cost) ** 2
        if isinstance(new_obj, WallObj) and len(new_obj.aruco_orients) < 2 \
           and not new_obj.is_wall_aligned(old_obj):
            cost = np.inf
        else:
            cost = ((new_obj.pose.x - old_obj.pose.x)**2 + (new_obj.pose.y - old_obj.pose.y)**2)
        return cost

    def max_association_cost(self, otype):
        if self.robot.particle_filter and \
           self.robot.particle_filter.state != self.robot.particle_filter.LOCALIZED:
            return np.inf
        if otype in (ArucoMarkerObj, WallObj, DoorwayObj):
            return np.inf  # should adjust based on pf uncertainty
        if otype is DominoObj:
            return DOMINO_MAX_ASSOCIATION_COST
        return 500  # should adjust based on pf uncertainty

    def pending_cost_threshold(self, candidate):
        if isinstance(candidate, DominoObj):
            return DOMINO_PENDING_COST_THRESHOLD
        return 50

    def associate_objects_of_type(self, otype):
        new = [c for c in self.candidates if type(c) is otype]
        old = [o for o in self.objects.values() if type(o) is otype]
        N_new = len(new)
        N_old = len(old)
        if N_old == 0:
            return
        costs = np.zeros([N_new,N_old])
        MAX_ACCEPTABLE_COST = self.max_association_cost(otype)
        for i in range(N_new):
            for j in range(N_old):
                if otype is ArucoMarkerObj and new[i].marker_id != old[j].marker_id:
                    costs[i,j] = MAX_ACCEPTABLE_COST + 1
                elif otype is AprilTagObj and new[i].tag_id != old[j].tag_id:
                    costs[i,j] = MAX_ACCEPTABLE_COST + 1
                else:
                    costs[i,j] = self.association_cost(new[i], old[j])
        # *** Greedy algorithm; replace with the Hungarian algorithm
        for i in range(N_new):
            bestj = costs[i,:].argmin()
            if costs[i,bestj] < MAX_ACCEPTABLE_COST:
                new[i].matched = old[bestj]
                costs[:,bestj] = 1 + MAX_ACCEPTABLE_COST
        #print(f'{old=}  {new=}  {costs=}  {new[0].matched=}')

    def update_associated_objects(self):
        for candidate in self.candidates:
            if candidate.matched:
                candidate.update_matched_object(self.robot)
                self.updated_objects.append(candidate.matched)
                candidate.matched.is_missing = False
                if candidate.matched in self.missing_objects:
                    self.missing_objects.remove(candidate.matched)

    def should_be_visible(self, obj):
        # Really crude approach for now.  Should be doing camera
        # projection and accounting for occlusion.  In the future we
        # should employ the depth map for occusion detection.
        # For now, just return true if the object's bearing is within
        # the camera field of view and the distance is not too large.
        dx = obj.pose.x - self.robot.pose.x
        dy = obj.pose.y - self.robot.pose.y
        bearing = wrap_angle(atan2(dy,dx) - self.robot.pose.theta)
        distance = (dx**2 + dy**2) ** 0.5
        DISTANCE_THRESHOLD = 400 # mm
        BEARING_THRESHOLD = 30 # degrees
        result = abs(bearing)*180/pi < BEARING_THRESHOLD and distance < DISTANCE_THRESHOLD
        return result

    def detect_missing_objects(self):
        for obj in self.objects.values():
            if not isinstance(obj, (ArucoMarkerObj,WallObj,DoorwayObj)) and \
               obj not in self.updated_objects and self.should_be_visible(obj):
                if obj not in self.missing_objects:
                    obj.is_visible = False
                    obj.is_missing = True
                    self.missing_objects.append(obj)
                    #print(f'missing object: {obj}, visibility_paused={self.visibility_paused} {self.updated_objects=}')

    def process_unassociated_objects(self):
        """
        The vision system produces lots of spurious objects, so we require
        a new object to be seen 6 times in successive camera frames before
        we add it to the world map.
        """
        unassociated = [c for c in self.candidates if c.matched is None]
        pending = list(self.pending_objects.keys())
        if self.robot.particle_filter and \
           self.robot.particle_filter.state != self.robot.particle_filter.LOCALIZED:
            pass # return
        if self.robot.particle_filter:
            pass # print('robot.particle_filter.state=', self.robot.particle_filter.state)
        for candidate in unassociated:
            if isinstance(candidate, WallObj) and len(candidate.aruco_orients) == 1:
                #print('punting on', candidate, 'from', self.objects)
                # only one aruco isn't enough to make a new wall
                continue
            cost_threshold = self.pending_cost_threshold(candidate)
            matches = [p for p in pending if self.association_cost(candidate,p) < cost_threshold]
            if matches:
                m = matches[0]
                self.pending_objects[m] += 1
                if self.pending_objects[m] >= 6:
                    if self.reclaim_object(candidate):
                        pass
                    else:
                        candidate.id = self.next_in_sequence(candidate.name)
                        candidate.pose = PoseEstimate(candidate.pose)
                        if isinstance(candidate, DominoObj) and hasattr(candidate.pose, 'kf_theta'):
                            candidate.pose.theta = normalize_axis_angle(candidate.pose.theta)
                            candidate.pose.kf_theta.state = candidate.pose.theta
                        self.objects[candidate.id] = candidate
                        candidate.is_visible = True
                        print('Added', candidate)
                        self.updated_objects.append(candidate)
                    del self.pending_objects[m]
                pending.remove(m)
            else:
                self.pending_objects[candidate] = 1
        for p in pending:
            #print('retracted', p, '  count=', self.pending_objects[p])
            del self.pending_objects[p]

    def reclaim_object(self, obj):
        t = type(obj)
        missing = [m for m in self.missing_objects if type(m) == t]
        if hasattr(obj,'marker_id'):
            missing = [m for m in missing if m.marker_id == obj.marker_id]
        if hasattr(obj,'tag_id'):
            missing = [m for m in missing if m.tag_id == obj.tag_id]
        if len(missing) == 0:
            return None
        costs = [self.association_cost(obj, m) for m in missing]
        min_index = np.argmin(costs)
        match = missing[min_index]
        match.is_visible = True
        match.is_missing = False
        match.pose = PoseEstimate(obj.pose)
        if isinstance(match, DominoObj):
            if hasattr(match.pose, 'kf_theta'):
                match.pose.theta = normalize_axis_angle(match.pose.theta)
                match.pose.kf_theta.state = match.pose.theta
            for attr in ('image_center', 'image_quad', 'image_axis', 'image_divider',
                         'half_counts', 'half_image_centers', 'half_world_centers',
                         'mask_area', 'face_label', 'face_confidence', 'confidence'):
                setattr(match, attr, getattr(obj, attr, None))
            match.is_fallen = getattr(obj, 'is_fallen', False)
        self.updated_objects.append(match)
        self.missing_objects.remove(match)
        #print('reclaimed', match)
        return match
        
    def next_in_sequence(self,name):
        count = 1 + self.name_counts.get(name, 0)
        self.name_counts[name] = count
        return name + "." + self.to_base_26(count)

    def to_base_26(self, num):
        result = []
        while num > 0:
            num -= 1  # Adjust for 1-based indexing (A=1, Z=26)
            remainder = num % 26
            result.append(chr(remainder + ord('a')))
            num //= 26
            return ''.join(reversed(result))
    
    def update_visibilities(self):
        for obj in self.objects.values():
            if obj not in self.updated_objects:
                obj.is_visible = False

    def update_holding(self):
        if self.robot.holding:
            self.confirm_still_holding()
        else:
            self.confirm_not_holding()

    def confirm_still_holding(self):
        MIN_UNHOLDING_TIME = 0.75  # seconds
        t = time.time()
        if (isinstance(self.robot.holding, BarrelObj) and self.robot.robot0.has_any_barrel()) or \
            (isinstance(self.robot.holding, SportsBallObj) and self.robot.robot0.has_sports_ball()):
            self.last_held_time = t
        else:
            if t - self.last_held_time > MIN_UNHOLDING_TIME:
                # held object has been gone long enough
                print('No longer holding', self.robot.holding)
                self.robot.holding.held_by = None
                self.robot.holding = None
            else:
                pass # wait a bit to see if held object comes back

    def check_spec_indicates_held(self,obj):
        if not isinstance(obj, (BarrelObj,SportsBallObj)):
            return False
        if isinstance(obj, BarrelObj):
            max_left = 150
            min_right = 460
        else:
            max_left = 120
            min_right = 480
        spec = obj.spec
        #print(f"left={spec['originx']*AIVISION_RESOLUTION_SCALE}  {max_left=}  " + \
        #      f"right={(spec['originx'] + spec['width']) * AIVISION_RESOLUTION_SCALE} {min_right=}")
        #
        # never seen objects acquired from map layout won't have originx
        if 'originx' in spec and \
           spec['originx']*AIVISION_RESOLUTION_SCALE < max_left and \
           (spec['originx'] + spec['width']) * AIVISION_RESOLUTION_SCALE > min_right:
            return True
        else:
            return False

    def confirm_not_holding(self):
        if self.robot.robot0.has_any_barrel() or self.robot.robot0.has_sports_ball():
            held_obj = None
            for obj in self.objects.values():
                if self.check_spec_indicates_held(obj):
                    held_obj = obj
                    break
            if held_obj:
                print('Robot now holding', held_obj)
                self.robot.holding = held_obj
                held_obj.held_by = self.robot
            else:
                pass # print('*** Could not find held object.')

    def reposition_held_object(self, obj):
        r = self.robot.kine.body_diameter/2 + obj.diameter/2
        pt = aboutZ(self.robot.pose.theta).dot(point(r,0))
        obj.pose.x = self.robot.pose.x + pt[0,0]
        obj.pose.y = self.robot.pose.y + pt[1,0]

    def update_held_object(self):
        if self.robot.holding:
            self.reposition_held_object(self.robot.holding)

    def show_objects(self):
        with self._lock:
            objs = sorted(self.objects.items(), key=lambda x: x[0])
            if len(objs) == 0:
                print('No objects in the world map.\n')
                return
            width = max([len(x[0]) for x in objs])
            for obj in objs:
                print(f'{obj[0].rjust(width)}: {obj[1]}')
            print()



################ GPT interface ################

    def get_prompt(self):
        with self._lock:
            prompt = ''
            prompt += f'It is now {datetime.datetime.now().strftime("%B %d, %Y, %I:%M:%S %p")}.\n'
            if self.robot.particle_filter.state == self.robot.particle_filter.LOCALIZED:
                prompt += f'You are located at ({round(self.robot.pose.x)}, {round(self.robot.pose.y)})\n'
                prompt += f'Your heading is {round(self.robot.pose.theta*180/pi)} degrees\n'
            else:
                prompt += f'You are currently lost (not localized) and do not see any landmarks.\n'
            if self.robot.holding:
                prompt += f'You are currently holding {self.robot.holding.id}.\n'
            else:
                prompt += f'You are not currently holding anything.\n'
            prompt += f'Your battery level is {self.robot.battery_percentage} percent.\n'
            for (id,obj) in self.objects.items():
                if not obj.is_missing:
                    if obj.is_visible:
                        vis = "visible"
                    else:
                        vis = "not vislbie"
                    prompt += f'{id} is located at ({round(obj.pose.x)}, {round(obj.pose.y)}) ' + \
                        f'and is {vis}\n'
                else:
                    prompt += f'{id} is missing\n'

                if isinstance(obj, WallObj):
                    front_markers = []
                    back_markers = []
                    for marker_id, marker_info in obj.wall_spec.marker_specs.items():
                        if marker_info['side'] == 1:  # +1 means front side
                            front_markers.append(marker_id)
                        else:  # -1 means back side
                            back_markers.append(marker_id)
                    prompt += f'{obj.id} has markers {front_markers} on its front side and {back_markers} on its back side\n'   

                if isinstance(obj, DoorwayObj) and obj.wall:
                    prompt += f'{id} is part of {obj.wall.id}\n'
            prompt += self.domino_prompt_section()
            landmark_ids = list(self.robot.particle_filter.sensor_model.landmarks.keys())
            if landmark_ids:
                for id in landmark_ids:
                    prompt += f'{id} is a navigation landmark.'
            else:
                prompt += 'There are currently no navigation landmarks.'    
            return prompt

    def domino_prompt_section(self):
        """Describe the dominoes on the table for the language model.

        Prefers robot.domino_bridge, which sorts the tiles into a board and two
        hands. Without it, falls back to a flat list of what the camera sees, so
        this stays useful before any game has been set up.
        """
        dominoes = [obj for obj in self.objects.values()
                    if isinstance(obj, DominoObj) and not obj.is_missing]
        if not dominoes:
            return ''
        bridge = getattr(self.robot, 'domino_bridge', None)
        if bridge is not None:
            try:
                return bridge.describe(bridge.observe()) + '\n'
            except Exception as exc:
                print(f'*** Domino bridge failed while building prompt: {exc}')
        faces = ', '.join(f'[{d.pips[0]}|{d.pips[1]}]' if d.pips else '[?|?]'
                          for d in dominoes)
        return f'You can see {len(dominoes)} domino tile(s): {faces}\n'

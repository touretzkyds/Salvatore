import numpy as np
from math import pi, nan

from .geometry import wrap_angle, Quaternion

class LinearKalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, process_noise, measurement_noise):
        """
        Initialize the Linear Kalman Filter
        
        Parameters:
        - initial_state: Starting state estimate
        - initial_uncertainty: Initial uncertainty in the state estimate
        - process_noise: Standard deviation of the process noise
        - measurement_noise: Standard deviation of the measurement noise
        """
        # State estimate 
        self.state = initial_state
        
        # Estimate uncertainty
        self.uncertainty = initial_uncertainty
        
        # Process noise covariance
        self.Q = process_noise**2
        
        # Measurement noise covariance
        self.R = measurement_noise**2
    
    def set_measurement_noise(self, measurement_noise):
        self.R = measurement_noise**2

    def predict(self, control_input=0):
        # Predict state (with optional control input)
        self.state += control_input
        
        # Increase uncertainty
        self.uncertainty += self.Q
        
        return self.state
    
    def update(self, measurement):
        # Calculate innovation (measurement residual)
        innovation = measurement - self.state
        
        # Calculate innovation covariance
        S = self.uncertainty + self.R
        
        # Calculate Kalman gain
        K = self.uncertainty / S
        
        # Update state estimate
        self.state += K * innovation
        
        # Update uncertainty
        self.uncertainty *= (1 - K)
        
        return self.state
    
    def get_state(self):
        return self.state


class CircularKalmanFilter:
    def __init__(self, initial_angle, initial_uncertainty, process_noise, measurement_noise):
        """
        Initialize the Circular Kalman Filter
        
        Parameters:
        - initial_angle: Starting angle (in radians)
        - initial_uncertainty: Initial uncertainty in the angle estimate
        - process_noise: Standard deviation of the process noise
        - measurement_noise: Standard deviation of the measurement noise
        """
        # State estimate (angle)
        self.state = initial_angle
        
        # Estimate uncertainty
        self.uncertainty = initial_uncertainty
        
        # Process noise covariance
        self.Q = process_noise**2
        
        # Measurement noise covariance
        self.R = measurement_noise**2
    
    def normalize_angle(self, angle):
        """
        Normalize angle to be within [-π, π]
        
        Parameters:
        - angle: Input angle in radians
        
        Returns:
        - Normalized angle
        """
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def angular_difference(self, a, b):
        """
        Calculate the shortest angular difference between two angles
        
        Parameters:
        - a: First angle in radians
        - b: Second angle in radians
        
        Returns:
        - Shortest angular difference
        """
        diff = a - b
        return self.normalize_angle(diff)
    
    def predict(self, control_input=0):
        """
        Prediction step of the Kalman filter
        
        Parameters:
        - control_input: Optional control input affecting the state (default 0)
        
        Returns:
        - Predicted state
        """
        # Predict state (with optional control input)
        self.state = self.normalize_angle(self.state + control_input)
        
        # Increase uncertainty
        self.uncertainty += self.Q
        
        return self.state
    
    def update(self, measurement):
        """
        Update step of the Kalman filter
        
        Parameters:
        - measurement: New angle measurement in radians
        
        Returns:
        - Updated state estimate
        """
        # Normalize measurement and current state
        normalized_measurement = self.normalize_angle(measurement)
        normalized_state = self.normalize_angle(self.state)
        
        # Calculate innovation (measurement residual)
        innovation = self.angular_difference(normalized_measurement, normalized_state)
        
        # Calculate innovation covariance
        S = self.uncertainty + self.R
        
        # Calculate Kalman gain
        K = self.uncertainty / S
        
        # Update state estimate
        self.state = self.normalize_angle(normalized_state + K * innovation)
        
        # Update uncertainty
        self.uncertainty *= (1 - K)
        
        return self.state
    
    def get_state(self):
        """
        Get the current state estimate
        
        Returns:
        - Current angle estimate in radians
        """
        return self.state


class Pose():
    def __init__(self, x=0, y=0, z=0, theta=None, quaternion = None, origin_id=-1):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        if quaternion is not None:
            self.quaternion = quaternion
        elif theta is not None:
            self.quaternion = Quaternion(angle_z = theta)
        else:
            self.quaternion = Quaternion(angle_z = 0)
        self.origin_id = origin_id

    def __repr__(self):
        theta = f' theta={self.theta*180/pi : .1f} deg.' if self.theta else ''
        return f'<Pose x={self.x:.1f} y={self.y:.1f} z={self.z:.1f}{theta} origin_id={self.origin_id}>'

    def __sub__(self, other):
        angdiff = wrap_angle(self.theta - other.theta) if self.theta is not None and other.theta is not None else None
        return Pose(self.x - other.x,
                    self.y - other.y,
                    self.z - other.z,
                    angdiff)
    
    def is_comparable(self, other):
        return self.origin_id == other.origin_id
    

class PoseEstimate(Pose):
    def __init__(self, x=0, y=0, z=0, theta=None, quaternion=None):
        if isinstance(x, Pose):
            p = x
            x = p.x
            y = p.y
            z = p.z
            theta = p.theta
            quaternion = p.quaternion
        super().__init__(x, y, z, theta, quaternion)
        initial_uncertainty = 200
        process_noise = 0.01
        base_measurement_noise = 0.1
        self.kf_x = LinearKalmanFilter(x, initial_uncertainty, process_noise, base_measurement_noise)
        self.kf_y = LinearKalmanFilter(y, initial_uncertainty, process_noise, base_measurement_noise)
        self.kf_z = LinearKalmanFilter(z, initial_uncertainty, process_noise, base_measurement_noise)
        theta_initial_uncertainty = 200
        theta_process_noise = 0.05
        theta_base_measurement_noise = 0.5
        if theta is not None:
            self.kf_theta = CircularKalmanFilter(theta,
                                                 theta_initial_uncertainty,
                                                 theta_process_noise,
                                                 theta_base_measurement_noise)

    def update(self, new_pose, measurement_noise=None):
        if measurement_noise is not None:
            self.kf_x.set_measurement_noise(measurement_noise)
            self.kf_y.set_measurement_noise(measurement_noise)
            self.kf_z.set_measurement_noise(measurement_noise)

        #self.kf_x.predict()
        self.x = self.kf_x.update(new_pose.x)

        #self.kf_y.predict()
        self.y = self.kf_y.update(new_pose.y)

        #self.kf_z.predict()
        self.z = self.kf_z.update(new_pose.z)

        if self.theta is not None and self.theta is not nan:
            #self.kf_theta.predict()
            self.theta = self.kf_theta.update(new_pose.theta)

    def __repr__(self):
        theta = f' theta={self.theta*180/pi : .1f} deg.' if self.theta else ''
        return f'<PoseEstimate x={self.x:.1f} y={self.y:.1f} z={self.z:.1f}{theta} origin_id={self.origin_id}>'

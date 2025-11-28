import numpy as np

from numpy.typing import ArrayLike
from racetrack import RaceTrack

# =============================================================================
# Tuning Parameters
# =============================================================================

# Path blending: 0.0 = centerline only, 1.0 = raceline only
RACELINE_BLEND = 0.55

# Pure Pursuit parameters
LOOKAHEAD_BASE = 6.0
LOOKAHEAD_GAIN = 0.25

# Velocity PID controller gains
KP_VELOCITY = 5.0
KI_VELOCITY = 0.15
KD_VELOCITY = 0.1
VELOCITY_INTEGRAL_MAX = 15.0

# Steering PID controller gains  
KP_STEERING = 7.0
KI_STEERING = 0.2
KD_STEERING = 0.25
STEERING_INTEGRAL_MAX = 0.4

# Velocity planning
MAX_LATERAL_ACCEL = 18.5
MIN_VELOCITY = 22.0

# Curvature lookahead for velocity planning
CURVATURE_LOOKAHEAD_POINTS = 5


# =============================================================================
# Helper Functions
# =============================================================================

def find_closest_point_index(position: ArrayLike, path: ArrayLike) -> int:
    distances = np.linalg.norm(path[:, :2] - position[:2], axis=1)
    return np.argmin(distances)


def compute_curvature(p1: ArrayLike, p2: ArrayLike, p3: ArrayLike) -> float:
    # Side lengths
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    
    # Avoid division by zero
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return 0.0
    
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    area = abs(cross) / 2.0
    curvature = 4.0 * area / (a * b * c)
    return curvature


def compute_path_curvatures(path: ArrayLike) -> ArrayLike:
    n = len(path)
    curvatures = np.zeros(n)
    
    for i in range(n):
        p1 = path[(i - 1) % n, :2]
        p2 = path[i, :2]
        p3 = path[(i + 1) % n, :2]
        curvatures[i] = compute_curvature(p1, p2, p3)
    return curvatures


def get_lookahead_point(position: ArrayLike, path: ArrayLike, lookahead_dist: float, start_idx: int) -> tuple:
    n = len(path)
    accumulated_dist = 0.0
    idx = start_idx
    
    while accumulated_dist < lookahead_dist:
        next_idx = (idx + 1) % n
        segment_dist = np.linalg.norm(path[next_idx, :2] - path[idx, :2])
        accumulated_dist += segment_dist
        idx = next_idx
        
        if idx == start_idx:
            break
    
    return path[idx, :2], idx


def compute_path_heading(path: ArrayLike, idx: int) -> float:
    n = len(path)
    next_idx = (idx + 1) % n
    dx = path[next_idx, 0] - path[idx, 0]
    dy = path[next_idx, 1] - path[idx, 1]
    return np.arctan2(dy, dx)


def blend_paths(raceline: ArrayLike, centerline: ArrayLike, blend_factor: float) -> ArrayLike:
    # If paths have different lengths, interpolate the shorter one
    n_race = len(raceline)
    n_center = len(centerline)
    
    if n_race != n_center:
        n_target = max(n_race, n_center)
        
        # Interpolate raceline
        t_race = np.linspace(0, 1, n_race)
        t_target = np.linspace(0, 1, n_target)
        raceline_interp = np.zeros((n_target, 2))
        raceline_interp[:, 0] = np.interp(t_target, t_race, raceline[:, 0])
        raceline_interp[:, 1] = np.interp(t_target, t_race, raceline[:, 1])
        
        # Interpolate centerline
        t_center = np.linspace(0, 1, n_center)
        centerline_interp = np.zeros((n_target, 2))
        centerline_interp[:, 0] = np.interp(t_target, t_center, centerline[:, 0])
        centerline_interp[:, 1] = np.interp(t_target, t_center, centerline[:, 1])
        
        raceline = raceline_interp
        centerline = centerline_interp
    
    blended = blend_factor * raceline + (1.0 - blend_factor) * centerline
    return blended


# =============================================================================
# Controller Blocks
# =============================================================================

_blocks = None

def init_blocks(raceline):
    global _blocks
    _blocks = Blocks(raceline)

class Blocks:
    def __init__(self, raceline: ArrayLike = None):
        # Cache for computed values
        self.curvature_cache = None
        self.velocity_profile_cache = None
        self.blended_path_cache = None
        
        # Store the raceline and blend factor
        self.raceline = raceline
        
        # PID state for velocity controller (C1)
        self.velocity_integral = 0.0
        self.prev_velocity_error = None
        self.prev_velocity = None
        
        # PID state for steering controller (C2)
        self.steering_integral = 0.0
        self.prev_steering_error = None
    
    def _get_path(self, racetrack: RaceTrack) -> ArrayLike:        
        if self.blended_path_cache is not None:
            return self.blended_path_cache
        
        centerline = racetrack.centerline[:, :2]
        self.blended_path_cache = blend_paths(self.raceline, centerline, RACELINE_BLEND)
        return self.blended_path_cache
    
    def _get_curvatures(self, racetrack: RaceTrack) -> ArrayLike:
        path = self._get_path(racetrack)
        if self.curvature_cache is None:
            self.curvature_cache = compute_path_curvatures(path)
        return self.curvature_cache
    
    def _get_velocity_profile(self, racetrack: RaceTrack, parameters: ArrayLike) -> ArrayLike:
        if self.velocity_profile_cache is not None:
            return self.velocity_profile_cache
        
        path = self._get_path(racetrack)
        curvatures = self._get_curvatures(racetrack)
        n = len(curvatures)
        
        max_velocity = parameters[5]
        max_accel = parameters[10]
        
        # Forward pass: compute curvature-limited velocities
        velocity_profile = np.zeros(n)
        for i in range(n):
            if curvatures[i] > 1e-6:
                v_curvature = np.sqrt(MAX_LATERAL_ACCEL / curvatures[i])
                velocity_profile[i] = min(max_velocity, v_curvature)
            else:
                velocity_profile[i] = max_velocity
        
        velocity_profile = np.maximum(velocity_profile, MIN_VELOCITY)
        
        # Backwards pass: ensure we can brake in time
        for _ in range(2):
            for i in range(n - 1, -1, -1):
                next_idx = (i + 1) % n
                ds = np.linalg.norm(path[next_idx, :2] - path[i, :2])
                v_brake = np.sqrt(velocity_profile[next_idx]**2 + 2 * max_accel * ds)
                velocity_profile[i] = min(velocity_profile[i], v_brake)
        
        # Forward pass: ensure we can accelerate to target
        for i in range(n):
            next_idx = (i + 1) % n
            ds = np.linalg.norm(path[next_idx, :2] - path[i, :2])
            v_accel = np.sqrt(velocity_profile[i]**2 + 2 * max_accel * ds)
            velocity_profile[next_idx] = min(velocity_profile[next_idx], v_accel)
        
        self.velocity_profile_cache = velocity_profile
        return velocity_profile
    
    def s1(self, state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> float:
        position = state[:2]
        path = self._get_path(racetrack)
        
        velocity_profile = self._get_velocity_profile(racetrack, parameters)
        closest_idx = find_closest_point_index(position, path)
        
        # Look ahead and find minimum velocity in upcoming section
        n = len(velocity_profile)
        min_upcoming_velocity = velocity_profile[closest_idx]
        
        for i in range(1, CURVATURE_LOOKAHEAD_POINTS):
            idx = (closest_idx + i) % n
            if velocity_profile[idx] < min_upcoming_velocity:
                min_upcoming_velocity = velocity_profile[idx]
        
        velocity_ref = min(velocity_profile[closest_idx], min_upcoming_velocity * 1.2)
        
        return max(velocity_ref, MIN_VELOCITY)
    
    def c1(self, state: ArrayLike, velocity_ref: float) -> float:
        current_velocity = state[3]
        velocity_error = velocity_ref - current_velocity
        
        # Proportional term
        p_term = KP_VELOCITY * velocity_error
        
        # Integral term 
        self.velocity_integral += velocity_error
        self.velocity_integral = np.clip(
            self.velocity_integral, 
            -VELOCITY_INTEGRAL_MAX, 
            VELOCITY_INTEGRAL_MAX
        )
        i_term = KI_VELOCITY * self.velocity_integral
        
        # Derivative term
        if self.prev_velocity is not None:
            dv = current_velocity - self.prev_velocity
            d_term = -KD_VELOCITY * dv
        else:
            d_term = 0.0
        
        self.prev_velocity = current_velocity
        self.prev_velocity_error = velocity_error
        
        acceleration = p_term + i_term + d_term
        
        return acceleration
    
    def s2(self, state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> float:
        position = state[:2]
        current_heading = state[4]
        current_velocity = state[3]
        wheelbase = parameters[0]
        max_steering = parameters[4]
        path = self._get_path(racetrack)
        
        # Dynamic lookahead distance
        lookahead_dist = LOOKAHEAD_BASE + LOOKAHEAD_GAIN * current_velocity
        
        closest_idx = find_closest_point_index(position, path)
        lookahead_pos, lookahead_idx = get_lookahead_point(
            position, path, lookahead_dist, closest_idx
        )
        
        # Compute heading to lookahead point
        dx = lookahead_pos[0] - position[0]
        dy = lookahead_pos[1] - position[1]
        heading_to_lookahead = np.arctan2(dy, dx)
        
        alpha = heading_to_lookahead - current_heading
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        
        # Pure Pursuit steering angle formula
        if lookahead_dist > 0:
            steering_ref = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_dist)
        else:
            steering_ref = 0.0
        
        steering_ref = np.clip(steering_ref, -max_steering, max_steering)
        
        return steering_ref
    
    def c2(self, state: ArrayLike, steering_angle_ref: float) -> float:
        current_steering = state[2]
        steering_error = steering_angle_ref - current_steering
        
        # Normalize steering error to [-pi, pi]
        steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error))
        
        # Proportional term
        p_term = KP_STEERING * steering_error
        
        # Integral term
        self.steering_integral += steering_error
        self.steering_integral = np.clip(
            self.steering_integral,
            -STEERING_INTEGRAL_MAX,
            STEERING_INTEGRAL_MAX
        )
        i_term = KI_STEERING * self.steering_integral
        
        # Derivative term
        if self.prev_steering_error is not None:
            d_error = steering_error - self.prev_steering_error
            # Normalize derivative
            d_error = np.arctan2(np.sin(d_error), np.cos(d_error))
            d_term = KD_STEERING * d_error
        else:
            d_term = 0.0
        
        self.prev_steering_error = steering_error
        
        # PID output
        steering_velocity = p_term + i_term + d_term
        
        return steering_velocity

def lower_controller(
    state : ArrayLike, desired : ArrayLike, parameters : ArrayLike
) -> ArrayLike:
    assert(desired.shape == (2,))

    return np.array([_blocks.c2(state, desired[0]), _blocks.c1(state, desired[1])]).T

def controller(
    state : ArrayLike, parameters : ArrayLike, racetrack : RaceTrack
) -> ArrayLike:
    if _blocks is None:
        init_blocks(racetrack.raceline)
    return np.array([_blocks.s2(state, parameters, racetrack), _blocks.s1(state, parameters, racetrack)]).T
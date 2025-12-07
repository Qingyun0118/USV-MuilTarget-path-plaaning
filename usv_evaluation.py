"""
Multi-objective evaluation for USV path planning based on the paper:
"Multi-objective path planning for unmanned surface vehicle with currents effects"
(Yong Ma et al., ISA Transactions 75, 2018)

This module provides functions to compute four objectives:
1. Path length (L0)
2. Path smoothness (theta0)
3. Economic cost (E0)
4. Safety (D_Safe)
"""

import numpy as np
import math
from scipy.interpolate import griddata

# Import constants from my_USV_Env (avoid circular import)
try:
    from my_USV_Env import OBSTACLE_POS_R, TARGET_POS, MAX, X_MIN, Y_MIN, X_MAX, Y_MAX, ROU_C, ROU_S
except ImportError:
    # Fallback defaults (should match my_USV_Env)
    OBSTACLE_POS_R = [
        np.array([7.0, 16.0, 4.5]),
        np.array([60.0, 10.0, 10.0]),
        np.array([90.0, 10.0, 10.0]),
        np.array([40.0, 4.0, 4.0]),
        np.array([90.0, 90.0, 5.0]),
        np.array([45.0, 65.0, 5.0]),
        np.array([42.0, 47.0, 4.0]),
        np.array([92.0, 64.0, 4.0]),
        np.array([86.0, 35.0, 4.0]),
    ]
    TARGET_POS = [
        np.array([80.0, 50.0]),
        np.array([40.0, 90.0]),
        np.array([10.0, 50.0]),
    ]
    MAX = 100.0
    X_MIN = 0.0
    Y_MIN = 0.0
    X_MAX = 100.0
    Y_MAX = 100.0
    ROU_C = 1.5
    ROU_S = 3.5

# USV length (meters) as per paper
L_USV = 92.6  # meters, but environment uses units of 0-100? Assume same scale.
# In the paper, length unit is nautical mile, but our environment uses arbitrary units (0-100).
# We'll keep the same scaling as environment (1 unit = 1 meter? Not sure).
# For safety distance calculations, we need consistent units.
# Since obstacles radii are given in same units, we assume units are meters.
# However, the environment's MAX=100 suggests it's a normalized space.
# We'll treat distances as dimensionless for safety calculation, but we need d_safe_min and d_safe_max.
# According to paper, d_safe_min = L_USV + r_ccj, d_safe_max = 3*L_USV + 1.8*r_ccj.
# We'll compute using the same units as obstacle radii (which are in environment units).
# We'll assume L_USV = 10.0 (as a guess) because obstacle radii are around 4-10.
# Better to set L_USV as a parameter.

class MultiObjectiveEvaluator:
    """
    Evaluates a path (sequence of points) according to four objectives.
    """
    def __init__(self, env=None, usv_length=10.0, usv_speed=6.0, fuel_consumption=1.0):
        """
        Args:
            env: MyUsvEnv instance (optional). If provided, uses its flow field and obstacles.
            usv_length: length of USV (in environment units) for safety calculation.
            usv_speed: constant speed of USV relative to water (V) in environment units per second.
            fuel_consumption: FCPUT (fuel consumption per unit time) in kg/h.
        """
        self.env = env
        self.L_USV = usv_length
        self.V = usv_speed
        self.FCPUT = fuel_consumption
        
        # Obstacles as circles: (x, y, radius)
        self.obstacles = [obs[:3] for obs in OBSTACLE_POS_R]  # each is [x, y, r]
        # Targets (not used in evaluation but may be needed)
        self.targets = TARGET_POS
        
    def compute_path_length(self, points):
        """
        Compute total path length L0 = sum of segment lengths.
        points: list of (x, y) coordinates.
        """
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(points)-1):
            p1 = np.array(points[i])
            p2 = np.array(points[i+1])
            total += np.linalg.norm(p2 - p1)
        return total
    
    def compute_smoothness(self, points):
        """
        Compute maximum turning angle theta0 = max(ζ_i) in degrees.
        ζ_i = abs(θ_{i+1} - θ_i) * 180 / π
        where θ_i = atan2(y_i - y_{i-1}, x_i - x_{i-1})
        """
        if len(points) < 3:
            return 0.0  # no turning angle
        thetas = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            theta = math.atan2(dy, dx)
            thetas.append(theta)
        max_zeta = 0.0
        for i in range(len(thetas)-1):
            delta = abs(thetas[i+1] - thetas[i])
            # Normalize delta to [0, pi]
            if delta > math.pi:
                delta = 2*math.pi - delta
            zeta = delta * 180.0 / math.pi
            if zeta > max_zeta:
                max_zeta = zeta
        return max_zeta
    
    def compute_economic_cost(self, points):
        """
        Compute economic cost E0 = sum over segments: s_i / (V + v_c) * FCPUT.
        where v_c is the current speed along the segment direction.
        If env is provided, uses flow velocity at segment midpoint.
        Otherwise assumes no currents (v_c = 0).
        """
        if len(points) < 2:
            return 0.0
        total_cost = 0.0
        for i in range(len(points)-1):
            p1 = np.array(points[i])
            p2 = np.array(points[i+1])
            s_i = np.linalg.norm(p2 - p1)
            # Compute current velocity component along segment direction
            if self.env is not None:
                # Get flow velocity at midpoint
                mid = (p1 + p2) / 2.0
                flow_u, flow_v = self.env.get_flow_velocity(mid)
                # Direction unit vector of segment
                dir_vec = (p2 - p1) / (s_i + 1e-8)
                # Project flow onto segment direction
                v_c = flow_u * dir_vec[0] + flow_v * dir_vec[1]
            else:
                v_c = 0.0
            # Ground speed = V + v_c (scalar sum? Actually V is USV speed relative to water,
            # and v_c is current component along direction. The effective speed along segment is V + v_c.
            # This matches paper's formula (16) where they use V + v_c (scalar sum).
            effective_speed = self.V + v_c
            if effective_speed <= 0:
                # Should not happen per constraint (8), but handle gracefully
                effective_speed = 1e-8
            # Travel time for this segment
            t_i = s_i / effective_speed
            # Energy consumption = t_i * FCPUT
            e_i = t_i * self.FCPUT
            total_cost += e_i
        return total_cost
    
    def compute_safety(self, points):
        """
        Compute safety degree D_Safe = sum over obstacles D_safej.
        D_safej defined by equation (12) in paper.
        """
        if len(points) < 2:
            return 0.0
        total_safety = 0.0
        for obs in self.obstacles:
            cx, cy, r = obs[0], obs[1], obs[2]
            # Compute minimum distance from path to obstacle center
            min_dist = self._min_distance_to_segments(points, (cx, cy))
            # Safety distances
            d_safe_min = self.L_USV + r
            d_safe_max = 3 * self.L_USV + 1.8 * r
            # Compute D_safej
            if min_dist <= d_safe_min:
                D_safej = 1.0
            elif min_dist >= d_safe_max:
                D_safej = 0.0
            else:
                D_safej = (min_dist - d_safe_min) / (d_safe_max - d_safe_min)
            total_safety += D_safej
        return total_safety
    
    def _min_distance_to_segments(self, points, target):
        """
        Compute minimum Euclidean distance from target point to any segment of the polyline.
        """
        tx, ty = target
        min_dist = float('inf')
        for i in range(len(points)-1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            # Distance from point to line segment
            dist = self._point_to_segment_distance(tx, ty, x1, y1, x2, y2)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    @staticmethod
    def _point_to_segment_distance(px, py, x1, y1, x2, y2):
        """
        Distance from point (px, py) to segment (x1,y1)-(x2,y2).
        """
        # Vector from segment start to end
        v = np.array([x2 - x1, y2 - y1])
        w = np.array([px - x1, py - y1])
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(w)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(np.array([px - x2, py - y2]))
        b = c1 / c2
        pb = np.array([x1 + b * v[0], y1 + b * v[1]])
        return np.linalg.norm(np.array([px, py]) - pb)
    
    def evaluate(self, points):
        """
        Compute all four objectives for given path.
        Returns dict with keys: 'path_length', 'smoothness', 'economic_cost', 'safety'.
        """
        L0 = self.compute_path_length(points)
        theta0 = self.compute_smoothness(points)
        E0 = self.compute_economic_cost(points)
        D_Safe = self.compute_safety(points)
        return {
            'path_length': L0,
            'smoothness': theta0,
            'economic_cost': E0,
            'safety': D_Safe
        }


# Convenience function for quick evaluation using default evaluator
def evaluate_path(points, env=None):
    """
    Quick evaluation using default parameters.
    """
    evaluator = MultiObjectiveEvaluator(env=env)
    return evaluator.evaluate(points)


if __name__ == '__main__':
    # Simple test
    test_points = [(5.0, 5.0), (50.0, 50.0), (80.0, 80.0)]
    result = evaluate_path(test_points)
    print("Test evaluation:", result)
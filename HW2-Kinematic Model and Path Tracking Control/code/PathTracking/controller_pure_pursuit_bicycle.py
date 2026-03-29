import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.3.1: Tune Pure Pursuit Gain
                 kp=0.1, Lfc=2.0):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.dt = model.dt
        self.l = model.l
        self.current_idx = 0
        #NewFeature: Added a steering saturation limit so the pure pursuit command stays within the bicycle model constraint.
        self.delta_limit = 40.0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    # State: [x, y, yaw, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        #NewFeature: Use a speed-aware, lower-bounded lookahead distance to improve pure pursuit robustness across track sections.
        Ld = max(self.kp * abs(v) + self.Lfc, 1e-3)
        
        # TODO 4.3.1: Pure Pursuit Control for Bicycle Kinematic Model
        target_idx = min_idx
        while target_idx < len(self.path) - 1:
            dx = self.path[target_idx, 0] - x
            dy = self.path[target_idx, 1] - y
            if np.hypot(dx, dy) >= Ld:
                break
            target_idx += 1

        self.current_idx = target_idx
        target = self.path[target_idx]
        alpha = utils.angle_norm(np.rad2deg(np.arctan2(target[1] - y, target[0] - x)) - yaw)
        next_delta = np.rad2deg(np.arctan2(2.0 * self.l * np.sin(np.deg2rad(alpha)), Ld))
        next_delta = np.clip(next_delta, -self.delta_limit, self.delta_limit)
        # [end] TODO 4.3.1
        return next_delta

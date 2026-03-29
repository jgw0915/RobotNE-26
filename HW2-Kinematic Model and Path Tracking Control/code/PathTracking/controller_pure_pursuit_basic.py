import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBasic(Controller):
    def __init__(self, model, 
                 # Optional TODO: Tune Pure Pursuit Gain
                 kp=0.5, Lfc=5):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.current_idx = 0

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0

        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        
        #NewFeature: Use a speed-aware, lower-bounded lookahead distance to keep pure pursuit stable at low and high speeds.
        Ld = max(self.kp * abs(v) + self.Lfc, 1e-3)

        # Optional TODO: Pure Pursuit Control for Basic Kinematic Model
        # You can implement this if you want to use Pure Pursuit for basic kinematic model in F1 Challenge
        target_idx = min_idx
        while target_idx < len(self.path) - 1:
            dx = self.path[target_idx, 0] - x
            dy = self.path[target_idx, 1] - y
            if np.hypot(dx, dy) >= Ld:
                break
            target_idx += 1

        self.current_idx = target_idx
        target = self.path[target_idx]
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        alpha = utils.angle_norm(theta_target - yaw)
        next_w = np.rad2deg(2.0 * v * np.sin(np.deg2rad(alpha)) / Ld)
        
        return next_w

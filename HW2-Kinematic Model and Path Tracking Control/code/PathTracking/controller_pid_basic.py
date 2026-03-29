import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBasic(Controller):
    def __init__(self,
                 model,
                 # TODO 4.1.2: Tune PID Gains
                 kp=2.2,
                 ki=0.015,
                 kd=0.35):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
        self.current_idx = 0
        #NewFeature: Added guard/tuning parameters for preview-based PID tracking stability on high-curvature F1 tracks.
        self.integral_limit = 8.0
        self.w_limit = 180.0
        self.lookahead_gain = 0.18
        self.lookahead_base = 4.5
        self.lookahead_max = 18.0
        self.heading_weight = 1.1
    
    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
        self.current_idx = 0
    
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        #NewFeature: Adding velocity info
        # Extract State
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx]

        # Problem 4.1.1
        # NewFeature : dded a preview-based PID enhancement with speed-aware lookahead, integral clamping, and angular-velocity saturation to improve lateral tracking stability and accuracy.
        preview_dist = np.clip(
            self.lookahead_gain * abs(v) + self.lookahead_base,
            self.lookahead_base,
            self.lookahead_max,
        )
        target_idx = min_idx
        while target_idx < len(self.path) - 1:
            dx_target = self.path[target_idx, 0] - x
            dy_target = self.path[target_idx, 1] - y
            if np.hypot(dx_target, dy_target) >= preview_dist:
                break
            target_idx += 1

        preview_target = self.path[target_idx]

        dx = x - target[0]
        dy = y - target[1]
        path_heading = np.deg2rad(target[2])
        path_left_normal = np.array([-np.sin(path_heading), np.cos(path_heading)])
        cte = dx * path_left_normal[0] + dy * path_left_normal[1]

        theta_preview = np.rad2deg(np.arctan2(preview_target[1] - y, preview_target[0] - x))
        theta_err = utils.angle_norm(theta_preview - yaw)
        preview_err = np.sin(np.deg2rad(theta_err))
        err = cte + self.heading_weight * preview_dist * preview_err

        if abs(err) < 4.0 and abs(theta_err) < 60.0:
            self.acc_ep += err * self.dt
            self.acc_ep = np.clip(self.acc_ep, -self.integral_limit, self.integral_limit)
        else:
            self.acc_ep *= 0.85

        w_ff = np.rad2deg(
            2.0 * max(abs(v), 1.0) * preview_err / max(preview_dist, 1e-3)
        )
        next_w = w_ff + self.kp * err + self.ki * self.acc_ep + self.kd * (err - self.last_ep) / self.dt
        self.last_ep = err
        next_w = np.clip(next_w, -self.w_limit, self.w_limit)

        return next_w


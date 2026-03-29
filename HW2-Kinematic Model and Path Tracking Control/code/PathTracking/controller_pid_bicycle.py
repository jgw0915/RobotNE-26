import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBicycle(Controller):
    def __init__(self, model, 
                 # TODO 4.1.3: Tune PID Gains
                 kp=1.2, 
                 ki=0.0, 
                 kd=0.2):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.dt = model.dt
        self.current_idx = 0
        #NewFeature: Added steering/output and integral limits to keep the tuned PID controller stable on sharp turns.
        self.delta_limit = 40.0
        self.integral_limit = 10.0
    
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
        
        # Extract State
        x, y, yaw = info["x"], info["y"], info["yaw"]

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 5:
            return 0.0

        # Search Nearest Target Locally
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        # New Feature: Added target speed reference to the path for better PID tuning and performance on F1 tracks with varying speed requirements.
        target = self.path[min_idx]
        
        # TODO 4.1.3: PID Control for Bicycle Kinematic Model
        theta_target = np.rad2deg(np.arctan2(target[1] - y, target[0] - x))
        theta_err = utils.angle_norm(theta_target - yaw)
        dist = np.sqrt(min_dist)
        err = dist * np.sin(np.deg2rad(theta_err))
        self.acc_ep += err * self.dt
        self.acc_ep = np.clip(self.acc_ep, -self.integral_limit, self.integral_limit)
        next_delta = self.kp * err + self.ki * self.acc_ep + self.kd * (err - self.last_ep) / self.dt
        self.last_ep = err
        next_delta = np.clip(next_delta, -self.delta_limit, self.delta_limit)
        # [end] TODO 4.1.3
        return next_delta

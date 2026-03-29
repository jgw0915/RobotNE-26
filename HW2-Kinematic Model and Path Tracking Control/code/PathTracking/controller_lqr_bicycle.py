import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, model, Q=None, R=None, control_state='steering_angle'):
        self.path = None
        if control_state == 'steering_angle':
            self.Q = np.eye(2)
            self.R = np.eye(1)
            # TODO 4.4.1: Tune LQR Gains
            self.Q[0,0] = 10
            self.Q[1,1] = 3
            self.R[0,0] = 1
        elif control_state == 'steering_angular_velocity':
            self.Q = np.eye(3)
            self.R = np.eye(1)
            # TODO 4.4.4: Tune LQR Gains
            self.Q[0,0] = 10
            self.Q[1,1] = 3
            self.Q[2,2] = 1
            self.R[0,0] = 1
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.dt = model.dt
        self.l = model.l
        self.control_state = control_state
        self.current_idx = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
        self.pdelta = 0
        self.current_idx = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None
        
        # Extract State 
        x, y, yaw, delta, v = info["x"], info["y"], info["yaw"], info["delta"], info["v"]
        yaw = utils.angle_norm(yaw)

        # Check if reached end of track
        if self.current_idx >= len(self.path) - 3:
            return 0.0
        
        #NewFeature: Switched to local nearest-point search and normalized the copied target yaw to keep LQR tracking progressing forward smoothly.
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest_local(self.path, (x,y), self.current_idx, lookahead=50)
        self.current_idx = min_idx
        target = self.path[min_idx].copy()
        target[2] = utils.angle_norm(target[2])
        
        if self.control_state == 'steering_angle':
            # TODO 4.4.1: LQR Control for Bicycle Kinematic Model with steering angle as control input
            theta_e = np.deg2rad(utils.angle_norm(target[2] - yaw))

            dx = x - target[0]
            dy = y - target[1]
            path_heading = np.deg2rad(target[2])
            path_left_normal = np.array([-np.sin(path_heading), np.cos(path_heading)])
            e = dx * path_left_normal[0] + dy * path_left_normal[1]

            A = np.array([
                [1.0, -v * self.dt],
                [0.0, 1.0]
            ])
            B = np.array([
                [0.0],
                [-(v / self.l) * self.dt]
            ])

            x_state = np.array([[e], [theta_e]])
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
            next_delta = np.rad2deg(float(-(K @ x_state)[0, 0]))
            next_delta = np.clip(next_delta, -40.0, 40.0)
            # [end] TODO 4.4.1
        elif self.control_state == 'steering_angular_velocity':
            # TODO 4.4.4: LQR Control for Bicycle Kinematic Model with steering angular velocity as control input
            theta_e = np.deg2rad(utils.angle_norm(target[2] - yaw))

            dx = x - target[0]
            dy = y - target[1]
            path_heading = np.deg2rad(target[2])
            path_left_normal = np.array([-np.sin(path_heading), np.cos(path_heading)])
            e = dx * path_left_normal[0] + dy * path_left_normal[1]
            delta_rad = np.deg2rad(delta)

            A = np.array([
                [1.0, -v * self.dt, 0.0],
                [0.0, 1.0, -(v / self.l) * self.dt],
                [0.0, 0.0, 1.0]
            ])
            B = np.array([
                [0.0],
                [0.0],
                [self.dt]
            ])

            x_state = np.array([[e], [theta_e], [delta_rad]])
            P = self._solve_DARE(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
            delta_dot = float(-(K @ x_state)[0, 0])
            next_delta = np.rad2deg(delta_rad + delta_dot * self.dt)
            next_delta = np.clip(next_delta, -40.0, 40.0)
            # [end] TODO 4.4.4
        
        return next_delta

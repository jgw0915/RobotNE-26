import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState
from Simulation.kinematic import KinematicModel

class KinematicModelBicycle(KinematicModel):
    def __init__(self,
            l = 30,     # distance between rear and front wheel
            dt = 0.05
        ):
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
        # TODO 2.3.1: Bicycle Kinematic Model
        delta = np.deg2rad(cstate.delta)
        v = state.v + cstate.a * self.dt
        w = np.rad2deg(v * np.tan(delta) / self.l)
        x = state.x + v * np.cos(np.deg2rad(state.yaw)) * self.dt
        y = state.y + v * np.sin(np.deg2rad(state.yaw)) * self.dt
        yaw = (state.yaw + w * self.dt) % 360
        # [end] TODO 2.3.1
        state_next = State(x, y, yaw, v, w)
        return state_next

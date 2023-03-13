import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        firstModel = ManiuplatorModel(Tp)
        firstModel.m3 = 0.1
        firstModel.r3 = 0.05
        # II:  m3=0.01, r3=0.01
        secondModel = ManiuplatorModel(Tp)
        secondModel.m3 = 0.01
        secondModel.r3 = 0.01
        # III: m3=1.0,  r3=0.3
        thirdModel = ManiuplatorModel(Tp)
        thirdModel.m3 = 1.0
        thirdModel.r3 = 0.3
        self.models = [firstModel, secondModel, thirdModel]
        self.i = 0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        pass

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u

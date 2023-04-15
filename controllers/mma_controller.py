import numpy as np
from models.manipulator_model import ManiuplatorModel
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from .controller import Controller


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        firstModel = FeedbackLinearizationController(Tp, m3=0.1, r3=0.05)
        # II:  m3=0.01, r3=0.01
        secondModel = FeedbackLinearizationController(Tp, m3=0.01, r3=0.01)
        # III: m3=1.0,  r3=0.3
        thirdModel = FeedbackLinearizationController(Tp, m3=1.0, r3=0.3)
        self.models = [firstModel, secondModel, thirdModel]
        self.i = 0
        self.Tp = Tp
        self.prev_x = np.zeros(4)

    def choose_model(self, x, u):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        #Euler forward estimation
        x_dot = [ reg.model.x_dot(x, u) * self.Tp + x  for reg in self.models]
        error = [np.linalg.norm(x_hat - x) for x_hat in x_dot]
        self.i = error.index(min(error))


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        u = self.models[self.i].calculate_control(x, q_r, q_r_dot, q_r_ddot)
        self.choose_model(x, u)
        return u

import numpy as np
from models.manipulator_model import ManiuplatorModel
from controllers.feedback_linearization_controller import FeedbackLinearizationController
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
        self.controller = FeedbackLinearizationController(Tp)
        self.i = 0
        self.Kd = np.array([10.0, 0]).reshape(1,2)
        self.Kp = np.array([.2, 0]).reshape(1,2)
        

    def choose_model(self, x, u):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        error = []
        for model in self.models:
            y_hat = model.x_dot(x,u)
            error.append( x[:, np.newaxis] - y_hat )
        if np.argmin(error) < len(self.models) and self.i != np.argmin(error):
            self.i = np.argmin(error)
            print("{0:{fill}<10}{1}{0:{fill}<10}".format("#"," Model Selector ", fill='#'))
            print("{0:{fill}<10}{1}{2} {0:{fill}<10}".format("#","Choose model:=", self.i, fill='#'))
        error = []

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        q = x[:2]
        q_dot = x[2:]
        q, q_dot = np.hsplit(x[np.newaxis,:], 2)
        q = np.transpose(q)
        q_dot = np.transpose(q_dot)
        if q_r_ddot.shape != (2,1):
            v = q_r_ddot[:, np.newaxis]
        else:
            v = q_r_ddot
        # Add feedback
        v += self.Kd@( q_dot - q_r_dot[:, np.newaxis] ) + self.Kp @ (q - q_r[:, np.newaxis])
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v + C @ q_dot
        self.choose_model(x, u)
        return u

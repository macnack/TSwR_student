import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kd = np.array([100.0, 100.0]).reshape(1,2)
        self.Kp = np.array([100.0, 100.0]).reshape(1,2)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q = np.transpose(np.array([x[:2]]))
        q_dot = np.transpose(np.array([x[2:]]))
        if q_r_ddot.shape != (2,1):
            v = q_r_ddot[:, np.newaxis]
        else:
            v = q_r_ddot
        # Adding feedback
        v -= self.Kd@( q_dot - q_r_dot[:, np.newaxis] ) + self.Kp @ (q - q_r[:, np.newaxis])

        M = self.model.M(x)
        C = self.model.C(x)
        tau = M @ v + C @ q_dot
        return tau
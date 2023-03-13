import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kd = 0.2
        self.Kp = 10.0

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q, q_dot = np.hsplit(x[np.newaxis,:], 2)
        q = np.transpose(q)
        q_dot = np.transpose(q_dot)
        if q_r_ddot.shape != (2,1):
            v = q_r_ddot[:, np.newaxis]
        else:
            v = q_r_ddot
        # Adding feedback
        v += self.Kd*( q_dot - q_r_dot[:, np.newaxis] ) + self.Kp * (q - q_r[:, np.newaxis])

        M = self.model.M(x)
        C = self.model.C(x)
        tau = M @ v + C @ q_dot
        if False:
            print("tau:={shape}".format(shape=tau.shape))
            print("M:={shape}".format(shape=M.shape))
            print("v:={shape}".format(shape=v.shape))
            print("C:={shape}".format(shape=C.shape))
            print("q_dot:={shape}".format(shape=q_dot.shape))
        return tau
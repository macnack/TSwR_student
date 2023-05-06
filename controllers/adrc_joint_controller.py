import numpy as np
from observers.eso import ESO
from .controller import Controller
from .pd_controller import PDDecentralizedController
from models.manipulator_model import ManiuplatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.pd_control = PDDecentralizedController(self.kp, self.kd)
        self.model = ManiuplatorModel(Tp)
        A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                     [0.0, 0.0, 0.0]], dtype=np.float32)
        self.B = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
        L = np.array([[3*p, 3*np.power(p, 2), np.power(p, 3)]],
                     dtype=np.float32)
        W = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        self.eso = ESO(A, self.B*b, W, L.T, q0, Tp)

    def set_b(self, b):
        # TODO update self.b and B in ESO
        self.b = b
        self.eso.set_B(self.B * b)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, idx):
        # TODO implement ADRC
        q, _ = x  # q, qdot

        invM = self.model.M([q, _, 0, 0])
        invM = np.linalg.inv(invM)
        self.set_b(invM[idx, idx])
        # update ESO
        q_est, q_dot_est, f_est = self.eso.get_state()
        u_pd = self.pd_control.calculate_control(
            q_est, q_dot_est, q_d, q_d_dot, q_d_ddot)
        u = (u_pd - f_est) / self.b
        # u = u_pd
        self.eso.update(q, u)
        return u

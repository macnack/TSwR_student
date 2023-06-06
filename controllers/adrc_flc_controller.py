import numpy as np

from observers.eso import ESO
from .controller import Controller
# from models.ideal_model import IdealModel #almost
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        p1, p2 = p
        self.L = np.array([[3*p1, 0], [0, 3*p2], [3*p1**2, 0],
                          [0, 3*p2**2], [p1**3, 0], [0, p2**3]])
        W = np.eye(6)[:2, :]
        A = np.eye(6, k=2)
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.init_params(q0[:2], q0[2:])

    def init_params(self, q, q_dot):
        # TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)
        C = self.model.C(x)

        A = np.zeros((6, 6))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 4:6] = np.eye(2)
        A[2:4, 2:4] = -M_inv @ C

        B = np.zeros((6, 2))
        B[2:4, :] = M_inv

        self.eso.A = A
        self.eso.B = B

    def update_params(self, M, C):
        # TODO Implement procedure to set eso.A and eso.B
        M = np.linalg.inv(M)
        MC = -(M @ C)
        A = np.eye(6, k=2)
        A[2, 2] = MC[0, 0]
        A[2, 3] = MC[0, 1]
        A[3, 2] = MC[1, 0]
        A[3, 3] = MC[1, 1]

        B = np.zeros((6, 2))
        B[2, 0] = MC[0, 0]
        B[2, 1] = MC[0, 1]
        B[3, 0] = MC[1, 0]
        B[3, 1] = MC[1, 1]

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q1, q2, q1_dot, q2_dot = x  # q, qdot
        q = np.array([q1, q2])
        M = self.model.M(x)
        C = self.model.C(x)
        self.update_params(M, C)

        z_hat = self.eso.get_state()
        g_hat_dot = z_hat[2:4]
        f_est = z_hat[4:]

        e = q_d - q
        e_dot = q_d_dot - g_hat_dot
        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e
        u = M @ (v - f_est) + C @ g_hat_dot

        self.eso.update(q[:, np.newaxis], u[:, np.newaxis])
        return u

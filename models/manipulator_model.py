import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3 = 0, r3 = 0.05):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.0
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1. / 12. * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1. / 12. * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2
        self.alpha = self.I_1 + self.I_2 + self.I_3 + \
                     self.m1 * ( ( self.l1 / 2.0 ) ** 2 ) + \
                     self.m2 * ( self.l1 ** 2 + ( self.l2 / 2.0 ) ** 2) + \
                     self.m3 * ( self.l1 ** 2 + self.l2 ** 2)
        self.beta = self.m2 * self.l1 * ( self.l2 / 2.0 ) + \
                    self.m3 * self.l1 * self.l2
        self.gamma = self.I_2 + self.I_3 + \
                     self.m2 * (( self.l2 / 2.0 ) ** 2) + \
                     self.m3 * self.l2 ** 2

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        _, q2, _, _ = x
        m11 = self.alpha + 2 * self.beta * np.cos(q2)
        m12 = self.gamma + self.beta * np.cos(q2)
        m21 = self.gamma + self.beta * np.cos(q2)
        m22 = self.gamma
        return np.array( [ m11, m12, m21, m22] ).reshape(2,2)
    
    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        _, q2, q1_dot, q2_dot = x
        c11 = - self.beta * np.sin(q2) * q2_dot
        c12 = - self.beta * np.sin(q2) * (q1_dot + q2_dot)
        c21 = self.beta * np.sin(q2) * q1_dot
        c22 = 0
        return np.array( [c11, c12, c21, c22] ).reshape(2,2)
    
    def x_dot(self, x, u):
        invM = np.linalg.inv(self.M(x))
        zeros = np.zeros((2, 2), dtype=np.float32)
        A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1), np.concatenate([zeros, -invM @ self.C(x)], 1)], 0)
        b = np.concatenate([zeros, invM], 0)
        return A @ x[:, np.newaxis] + b @ u
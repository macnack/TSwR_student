from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        # TODO implement ESO update
        z_hat = self.state[:, np.newaxis]
        state_dot = np.zeros_like(self.A[0, :])
        if isinstance(u, (np.ndarray, np.float64)):
            if isinstance(u, np.ndarray):
                state_dot = self.A @ z_hat + self.B @ u + \
                    self.L @ (q - self.W @ z_hat)
            else:
                state_dot = self.A @ z_hat + self.B * \
                    u + self.L @ (q - self.W @ z_hat)
        self.state = self.state + self.Tp * \
            np.reshape(state_dot, (state_dot.shape[0], ))
        # shapes in numpy drives me crezy

    def get_state(self):
        return self.state

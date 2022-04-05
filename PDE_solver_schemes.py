import numpy as np


class crankNicholson:
    def __init__(self, f, mx, lmbda,x):
        self.f = f
        self.mx = mx
        self.lmbda = lmbda
        self.x = x

    def gen_tridiag(self):
        a = np.eye(self.mx, self.mx, k=-1) * - self.lmbda * 0.5 + np.eye(self.mx, self.mx) * (1 + self.lmbda) + \
            np.eye(self.mx, self.mx, k=1) * -self.lmbda * 0.5
        b = np.eye(self.mx, self.mx, k=-1) * self.lmbda * 0.5 + np.eye(self.mx, self.mx) * (1 - self.lmbda) + \
            np.eye(self.mx, self.mx, k=1) * self.lmbda * 0.5
        return a, b

    def u_j_mat(self):
        return np.zeros((self.mx, self.mx))

    def solve_cn(self,x):
        u_jp = self.u_j_mat()
        u_jp[:, 0] = self.u_i(x).T
        return


class forwardEuler:
    def __init__(self, f, mx, lmbda):
        self.f = f
        self.mx = mx
        self.lmbda = lmbda

    def tridiag_mat(self):
        # A function generating a tridiagonal (m-1) x (m-1) matrix
        return np.eye(self.mx + 1, self.mx + 1, k=-1) * self.lmbda + np.eye(self.mx + 1, self.mx + 1) * \
               (1 - 2 * self.lmbda) + np.eye(self.mx + 1, self.mx + 1, k=1) * self.lmbda

    def u_j_mat(self):
        return np.zeros((self.mx + 1, self.mx + 1))

    def solve(self,u_jp,x):
        u_jp = self.u_j_mat()
        u_jp[:, 0] = self.u_i(x).T
        for i in range(1, self.mx + 1):
            self.u_jp[:, i] = np.matmul(self.tridiag_mat(), u_jp[:, i - 1])
        return u_jp
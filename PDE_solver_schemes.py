import numpy as np
import pylab as pl
from math import pi


class forwardEuler:
    def __init__(self, u_j, mx, lmbda, mt):
        self.u_j = u_j
        self.mx = mx
        self.lmbda = lmbda
        self.mt = mt

    def tridiag_mat(self):
        # A function generating a tridiagonal (m-1) x (m-1) matrix
        return np.eye(self.mx + 1, self.mx + 1, k=-1) * self.lmbda + np.eye(self.mx + 1, self.mx + 1) * \
               (1 - 2 * self.lmbda) + np.eye(self.mx + 1, self.mx + 1, k=1) * self.lmbda

    def solve(self):
        for i in range(1, self.mt + 1):
            u_j1 = np.matmul(self.tridiag_mat(), self.u_j)
            u_j1[0] = 0
            u_j1[self.mx] = 0
            self.u_j[:] = u_j1[:]
        return


class backwardEuler:
    def __init__(self, u_j, mx, lmbda, mt):
        self.u_j = u_j
        self.mx = mx
        self.lmbda = lmbda
        self.mt = mt

    def tridiag_mat(self):
        # A function generating a tridiagonal (m-1) x (m-1) matrix for the backward Euler method
        return np.eye(self.mx + 1, self.mx + 1, k=-1) * -self.lmbda + np.eye(self.mx + 1, self.mx + 1) * \
               (1 + 2 * self.lmbda) + np.eye(self.mx + 1, self.mx + 1, k=1) * -self.lmbda

    def solve(self):
        for i in range(1, self.mt + 1):
            u_j1 = np.matmul(self.u_j, np.linalg.inv(self.tridiag_mat()))
            u_j1[0] = 0
            u_j1[self.mx] = 0
            self.u_j[:] = u_j1[:]
        return


class crankNicholson:
    def __init__(self, u_j, mx, lmbda, mt):
        self.u_j = u_j
        self.mx = mx
        self.lmbda = lmbda
        self.mt = mt
        self.a = None
        self.b = None

    def tridiag_mat(self):
        self.a = np.eye(self.mx+1, self.mx+1, k=-1) * - self.lmbda * 0.5 + np.eye(self.mx+1, self.mx+1) * (1 + self.lmbda) + \
                 np.eye(self.mx+1, self.mx+1, k=1) * -self.lmbda * 0.5
        self.b = np.eye(self.mx+1, self.mx+1, k=-1) * self.lmbda * 0.5 + np.eye(self.mx+1, self.mx+1) * (1 - self.lmbda) + \
                 np.eye(self.mx+1, self.mx+1, k=1) * self.lmbda * 0.5
        return

    def solve(self):
        crankNicholson.tridiag_mat(self)
        ab = np.matmul(self.b,np.linalg.inv(self.a))
        for i in range(1, self.mt + 1):
            u_j1 = np.matmul(self.u_j,ab)
            u_j1[0] = 0
            u_j1[self.mx] = 0
            self.u_j[:] = u_j1[:]
        return


if __name__ == '__main__':
    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant
    L = 1.0  # length of spatial domain
    T = 0.5  # total time to solve for

    # Set numerical parameters
    mx = 10  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time`

    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
    print("deltax=", deltax)
    print("deltat=", deltat)
    print("lambda=", lmbda)

    # Set up the solution variables
    u_j = np.sin(pi * x / L).T  # u at current time step

    solver = forwardEuler(u_j, mx, lmbda, mt)
    solver.solve()

    pl.plot(x, solver.u_j, 'ro', label='num')
    xx = np.linspace(0, L, 250)
    # pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()


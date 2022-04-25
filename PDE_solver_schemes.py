import numpy as np
import pylab as pl
from math import pi


class forwardEuler:
    def __init__(self, u_j, mx, lmbda, mt, p_func, q_func, bound_cond,deltax):
        self.u_j = u_j
        self.mx = mx
        self.lmbda = lmbda
        self.mt = mt
        self.p_func = p_func
        self.q_func = q_func
        self.bound_cond = bound_cond
        self.solution_matrix = self.gen_solution_mat()
        self.deltax = deltax

    def additive_vec(self):
        if self.bound_cond == 'dirichlet':
            return np.zeros(self.mx - 1)
        elif self.bound_cond == 'neumann':
            return np.zeros(self.mx+1)

    def tridiag_mat(self):
        # A function generating a tridiagonal (m-1) x (m-1) matrix
        if self.bound_cond == 'dirichlet':
            dims = self.mx-1
            return np.eye(dims, dims, k=-1) * self.lmbda + np.eye(dims, dims) * \
               (1 - 2 * self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda
        elif self.bound_cond == 'neumann':
            dims = self.mx+1
            mat  = np.eye(dims, dims, k=-1) * self.lmbda + np.eye(dims, dims) * \
               (1 - 2 * self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda
            mat[0, 1],mat[-1, -2] = 2 * mat[0, 1],2 * mat[-1, -2]
            return mat
        elif self.bound_cond == 'periodic':



    def gen_solution_mat(self):
        sol_mat = np.zeros((self.mt, self.mx + 1))
        sol_mat[0]= self.u_j
        return sol_mat



    def solve(self):
        # Check whether forward_euler method is suitable
        if not 0 < self.lmbda < 1 / 2:
            raise RuntimeError("Invalid value for lmbda")

        additive_vector = self.additive_vec()

        for i in range(0, self.mt - 1):
            if self.bound_cond == 'dirichlet':
                additive_vector[0], additive_vector[-1] = self.p_func(i), self.q_func(i)
                u_j1 = np.matmul(self.tridiag_mat(),  self.solution_matrix[i][1:-1]) + lmbda*additive_vector
                # setting boundary conditions
                self.solution_matrix[i+1][1:-1] = u_j1[:]
                self.solution_matrix[i + 1][0] = additive_vector[0]
                self.solution_matrix[i + 1][-1] = additive_vector[-1]

            elif self.bound_cond == 'neumann':
                additive_vector[0]= -self.p_func(i)
                additive_vector[-1]=  self.q_func(i)
                #additive_vector[0]= -additive_vector[0]
                u_j1 = np.matmul(self.tridiag_mat(), self.solution_matrix[i]) + 2*self.deltax*lmbda * additive_vector
                # setting boundary conditions
                self.solution_matrix[i + 1] = u_j1[:]

            elif self.bound_cond == 'periodic':






            elif self.bound_cond == 'homogenous':
                u_j1 = np.matmul(self.tridiag_mat(),self.solution_matrix[i][1:-1])
                # setting boundary conditions
                u_j1[0], u_j1[self.mx] = 0, 0
                self.solution_matrix[i+1][1:-1] = u_j1[:]

        print( self.solution_matrix)
        return  self.solution_matrix


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
        for i in range(1, self.mt - 1):
            u_j1 = np.matmul(self.u_j, np.linalg.inv(self.tridiag_mat()))
            u_j1[0], u_j1[self.mx] = 0, 0
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
        self.a = np.eye(self.mx + 1, self.mx + 1, k=-1) * - self.lmbda * 0.5 + np.eye(self.mx + 1, self.mx + 1) * (
                1 + self.lmbda) + \
                 np.eye(self.mx + 1, self.mx + 1, k=1) * -self.lmbda * 0.5
        self.b = np.eye(self.mx + 1, self.mx + 1, k=-1) * self.lmbda * 0.5 + np.eye(self.mx + 1, self.mx + 1) * (
                1 - self.lmbda) + \
                 np.eye(self.mx + 1, self.mx + 1, k=1) * self.lmbda * 0.5
        return

    def solve(self):
        crankNicholson.tridiag_mat(self)
        ab = np.matmul(self.b, np.linalg.inv(self.a))
        for i in range(1, self.mt + 1):
            u_j1 = np.matmul(self.u_j, ab)
            u_j1[0], u_j1[self.mx] = 0, 0
            self.u_j[:] = u_j1[:]

        return


def p(t):
    return 5


def q(t):
    return 3


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

    solver = forwardEuler(u_j, mx, lmbda, mt, p, q, 'neumann',deltax)
    solver.solve()

    pl.plot(x, solver.solution_matrix[-1], 'ro', label='num')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.show()

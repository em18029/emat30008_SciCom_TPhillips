import numpy as np
import pylab as pl
from math import pi


class forwardEuler:
    """
    A class which performs the forwardEuler method in matrix form.

     Parameters
    ----------
    u_j: 	numpy.array
        The solutions at t=1 of an PDE.
    mx: int
        An integer value for number of gridpoints in space.
    lmbda:	int
        The calculated mech fourier number.
    mt:	int
        An integer value for number of gridpoints in time.
    p_func:	function
        A function containing a float for the additive vector.
    q_func:	function
        A function containing a float for the additive vector.
    bound_cond:	string
        A string which determines what boundary conditions the solver will incorperate.
    solution_matrix: numpy.array
        A matrix for which the solver iteraitvly appends solutions at each timestep.

    Returns
    -------
    solution_matrix: numpy.array
        Array of solutions converging upon a steady state.
    """
    def __init__(self, u_j, mx, lmbda, mt, p_func, q_func, bound_cond, deltax):
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
        '''
        A function which returns the size an additive vector given the boundary conditions given.
        Returns: additive vector
        '''
        if self.bound_cond == 'dirichlet':
            return np.zeros(self.mx - 1)
        elif self.bound_cond == 'neumann':
            return np.zeros(self.mx + 1)
        else:
            return None

    def tridiag_mat(self):
        '''
        A function which returns a tridiagonal matrix using the numpy.eye(), its dimensions and other qualities are
        determined by the boundary conditions given.
        '''
        # A function generating a tridiagonal (m-1) x (m-1) matrix
        if self.bound_cond == 'dirichlet':
            dims = self.mx - 1
            return np.eye(dims, dims, k=-1) * self.lmbda + np.eye(dims, dims) * \
                   (1 - 2 * self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda
        elif self.bound_cond == 'neumann':
            dims = self.mx + 1
            mat = np.eye(dims, dims, k=-1) * self.lmbda + np.eye(dims, dims) * \
                  (1 - 2 * self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda
            mat[0, 1], mat[-1, -2] = 2 * mat[0, 1], 2 * mat[-1, -2]
            return mat
        elif self.bound_cond == 'periodic':
            dims = self.mx
            mat = np.eye(dims, dims, k=-1) * self.lmbda + np.eye(dims, dims) * \
                  (1 - 2 * self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda
            mat[0, -1], mat[-1, 0] = -self.lmbda, -self.lmbda
            return mat
        else:
            dims = self.mx - 1
            return np.eye(dims, dims, k=-1) * self.lmbda + np.eye(dims, dims) * \
                   (1 - 2 * self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda

    def gen_solution_mat(self):
        '''
        A function which returns a solution matrix of the right size method to be....
        '''
        if self.bound_cond == 'periodic':
            sol_mat = np.zeros((self.mt, self.mx))
            sol_mat[0] = self.u_j[:-1]
            return sol_mat
        else:
            sol_mat = np.zeros((self.mt, self.mx + 1))
            sol_mat[0] = self.u_j
            return sol_mat

    def solve(self):
        '''
        A function which utilises the forward euler method in matrix to solve a given pde.
        returns: np.array()
            matrix of solutions of dimensions (mx,mt)
        '''
        # Check whether forward_euler method is suitable
        if 0 < self.lmbda < 0.5 :
            raise RuntimeError("Invalid value for lmbda")

        a_vec = self.additive_vec()

        for i in range(0, self.mt - 1):
            if self.bound_cond == 'dirichlet':
                a_vec[0], a_vec[-1] = self.p_func(i), self.q_func(i)
                u_j1 = np.matmul(self.tridiag_mat(), self.solution_matrix[i][1:-1] + self.lmbda * a_vec)
                # setting boundary conditions
                self.solution_matrix[i + 1][1:-1] = u_j1[:]
                self.solution_matrix[i + 1][0] = a_vec[0]
                self.solution_matrix[i + 1][-1] = a_vec[-1]

            elif self.bound_cond == 'neumann':
                a_vec[0] = -self.p_func(i)
                a_vec[-1] = self.q_func(i)
                # a_vec[0]= -a_vec[0]
                u_j1 = np.matmul(self.tridiag_mat(),
                                 self.solution_matrix[i] + 2 * self.deltax * self.lmbda * a_vec)
                # setting boundary conditions
                self.solution_matrix[i + 1] = u_j1[:]

            elif self.bound_cond == 'periodic':

                self.solution_matrix[i + 1] = np.matmul(self.tridiag_mat(), self.solution_matrix[i])

            elif self.bound_cond == 'homogenous':
                self.solution_matrix[i + 1][1:-1] = np.matmul(self.tridiag_mat(), self.solution_matrix[i][1:-1])
                # setting boundary conditions
                # u_j1[0], u_j1[-1] = 0, 0
                # self.solution_matrix[i+1][1:-1] = u_j1[:]

        return self.solution_matrix


class backwardEuler:
    def __init__(self, u_j, mx, lmbda, mt, p_func, q_func, bound_cond, deltax):
        """
         A class which performs the backward Euler method in matrix form.

          Parameters
         ----------
         u_j: 	numpy.array
             The solutions at t=1 of an PDE.
         mx: int
             An integer value for number of gridpoints in space.
         lmbda:	int
             The calculated mech fourier number.
         mt:	int
             An integer value for number of gridpoints in time.
         p_func:	function
             A function containing a float for the additive vector.
         q_func:	function
             A function containing a float for the additive vector.
         bound_cond:	string
             A string which determines what boundary conditions the solver will incorperate.
         solution_matrix: numpy.array
             A matrix for which the solver iteraitvly appends solutions at each timestep.

         Returns
         -------
         solution_matrix: numpy.array
             Array of solutions converging upon a steady state.
         """
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
        '''
        A function which returns the size an additive vector given the boundary conditions given.
        Returns: np.array
            Additive vector
        '''
        if self.bound_cond == 'dirichlet':
            return np.zeros(self.mx - 1)
        elif self.bound_cond == 'neumann':
            return np.zeros(self.mx + 1)
        else:
            return None

    def tridiag_mat(self):
        '''
        A function which returns a tridiagonal matrix using the numpy.eye(), its dimensions and other qualities are
        determined by the boundary conditions given.
        '''
        if self.bound_cond == 'dirichlet':
            dims = self.mx - 1
            return np.eye(dims, dims, k=-1) * -self.lmbda + np.eye(dims, dims) * \
                   (1 + 2 * self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda
        elif self.bound_cond == 'neumann':
            dims = self.mx + 1
            mat = np.eye(dims, dims, k=-1) * -self.lmbda + np.eye(dims, dims) * \
                  (1 + 2 * self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda
            mat[0, 1], mat[-1, -2] = 2 * mat[0, 1], 2 * mat[-1, -2]
            return mat
        elif self.bound_cond == 'periodic':
            dims = self.mx
            mat = np.eye(dims, dims, k=-1) * -self.lmbda + np.eye(dims, dims) * \
                  (1 + 2 * self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda
            mat[0, -1], mat[-1, 0] = self.lmbda, self.lmbda
            return mat
        else:
            dims = self.mx - 1
            return np.eye(dims, dims, k=-1) * -self.lmbda + np.eye(dims, dims) * \
                   (1 + 2 * self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda

    def gen_solution_mat(self):
        '''
        A function which returns a solution matrix of the right size method to be....
        '''
        if self.bound_cond == 'periodic':
            sol_mat = np.zeros((self.mt, self.mx))
            sol_mat[0] = self.u_j[:-1]
            return sol_mat
        else:
            sol_mat = np.zeros((self.mt, self.mx + 1))
            sol_mat[0] = self.u_j
            return sol_mat

    def solve(self):
        '''
        A function which utilises the forward euler method in matrix to solve a given pde.
        returns: np.array()
            matrix of solutions of dimensions (mx,mt)
        '''
        a_vec = self.additive_vec()
        for i in range(0, self.mt - 1):
            if self.bound_cond == 'dirichlet':
                a_vec[0], a_vec[-1] = self.p_func(i), self.q_func(i)
                u_j1 = np.linalg.solve(self.tridiag_mat(), self.solution_matrix[i][1:-1] + self.lmbda * a_vec)
                # setting boundary conditions
                self.solution_matrix[i + 1][1:-1] = u_j1[:]
                self.solution_matrix[i + 1][0] = a_vec[0]
                self.solution_matrix[i + 1][-1] = a_vec[-1]

            elif self.bound_cond == 'neumann':
                a_vec[0] = -self.p_func(i)
                a_vec[-1] = self.q_func(i)
                # a_vec[0]= -a_vec[0]
                u_j1 = np.linalg.solve(self.tridiag_mat(),
                                       self.solution_matrix[i] + 2 * self.deltax * self.lmbda * a_vec)
                # setting boundary conditions
                self.solution_matrix[i + 1] = u_j1[:]

            elif self.bound_cond == 'periodic':
                self.solution_matrix[i + 1] = np.linalg.solve(self.tridiag_mat(), self.solution_matrix[i])

            elif self.bound_cond == 'homogenous':
                self.solution_matrix[i + 1][1:-1] = np.linalg.solve(self.tridiag_mat(), self.solution_matrix[i][1:-1])
                # setting boundary conditions
                # u_j1[0], u_j1[-1] = 0, 0
                # self.solution_matrix[i+1][1:-1] = u_j1[:]

        return self.solution_matrix


class crankNicholson:
    def __init__(self, u_j, mx, lmbda, mt, p_func, q_func, bound_cond, deltax):
        """
         A class which performs the Crank Nicholson method in matrix form.

          Parameters
         ----------
         u_j: 	numpy.array
             The solutions at t=1 of an PDE.
         mx: int
             An integer value for number of gridpoints in space.
         lmbda:	int
             The calculated mech fourier number.
         mt:	int
             An integer value for number of gridpoints in time.
         p_func:	function
             A function containing a float for the additive vector.
         q_func:	function
             A function containing a float for the additive vector.
         bound_cond:	string
             A string which determines what boundary conditions the solver will incorperate.
         solution_matrix: numpy.array
             A matrix for which the solver iteraitvly appends solutions at each timestep.

         Returns
         -------
         solution_matrix: numpy.array
             Array of solutions converging upon a steady state.
         """
        self.u_j = u_j
        self.mx = mx
        self.lmbda = lmbda
        self.mt = mt
        self.p_func = p_func
        self.q_func = q_func
        self.bound_cond = bound_cond
        self.solution_matrix = self.gen_solution_mat()
        self.deltax = deltax
        self.a = None
        self.b = None

    def additive_vec(self):
        '''
        A function which returns the size an additive vector given the boundary conditions given.
        Returns: np.array
            Additive vector
        '''
        if self.bound_cond == 'dirichlet':
            return np.zeros(self.mx - 1)
        elif self.bound_cond == 'neumann':
            return np.zeros(self.mx + 1)
        else:
            return None

    def tridiag_mat(self):
        '''
        A function which returns a tridiagonal matrix using the numpy.eye(), its dimensions and other qualities are
        determined by the boundary conditions given.
        '''
        if self.bound_cond == 'dirichlet':
            dims = self.mx - 1
            self.a = np.eye(dims, dims, k=-1) * -self.lmbda * 0.5 + np.eye(dims, dims) * \
                     (1 + self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda * 0.5
            self.b = np.eye(dims, dims, k=-1) * self.lmbda * 0.5 + np.eye(dims, dims) * \
                     (1 - self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda * 0.5

        elif self.bound_cond == 'neumann':
            dims = self.mx + 1
            self.a = np.eye(dims, dims, k=-1) * self.lmbda * -0.5 + np.eye(dims, dims) * \
                     (1 + self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda * 0.5
            self.b = np.eye(dims, dims, k=-1) * self.lmbda * 0.5 + np.eye(dims, dims) * \
                     (1 - self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda * 0.5

            self.a[0, 1], self.a[-1, -2] = 2 * self.a[0, 1], 2 * self.a[-1, -2]
            self.b[0, 1], self.b[-1, -2] = 2 * self.b[0, 1], 2 * self.b[-1, -2]
        elif self.bound_cond == 'periodic':
            dims = self.mx
            self.a = np.eye(dims, dims, k=-1) * -self.lmbda * 0.5 + np.eye(dims, dims) * \
                     (1 + self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda * 0.5
            self.b = np.eye(dims, dims, k=-1) * self.lmbda * 0.5 + np.eye(dims, dims) * \
                     (1 - self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda * 0.5

            self.a[0, -1], self.a[-1, 0] = self.lmbda * 0.5, self.lmbda * 0.5
            self.b[0, -1], self.b[-1, 0] = self.lmbda * 0.5, self.lmbda * 0.5
        else:
            dims = self.mx - 1
            self.a = np.eye(dims, dims, k=-1) * -self.lmbda * 0.5 + np.eye(dims, dims) * \
                     (1 + self.lmbda) + np.eye(dims, dims, k=1) * -self.lmbda * 0.5

            self.b = np.eye(dims, dims, k=-1) * self.lmbda * 0.5 + np.eye(dims, dims) * \
                     (1 - self.lmbda) + np.eye(dims, dims, k=1) * self.lmbda * 0.5

    def gen_solution_mat(self):
        '''
        A function which returns a solution matrix of the right size method to be....
        '''
        if self.bound_cond == 'periodic':
            sol_mat = np.zeros((self.mt, self.mx))
            sol_mat[0] = self.u_j[:-1]
            return sol_mat
        else:
            sol_mat = np.zeros((self.mt, self.mx + 1))
            sol_mat[0] = self.u_j
            return sol_mat

    def solve(self):
        '''
        A function which utilises the forward euler method in matrix to solve a given pde.
        returns: np.array()
            matrix of solutions of dimensions (mx,mt)
        '''
        a_vec = self.additive_vec()
        self.tridiag_mat()
        print(self.a)
        print(self.b)
        for i in range(0, self.mt - 1):
            if self.bound_cond == 'dirichlet':
                a_vec[0], a_vec[-1] = self.p_func(i), self.q_func(i)
                u_j1 = np.linalg.solve(self.a, np.dot(self.b, self.solution_matrix[i][1:-1]) + self.lmbda * a_vec)
                # setting boundary conditions
                self.solution_matrix[i + 1][1:-1] = u_j1[:]
                self.solution_matrix[i + 1][0] = a_vec[0]
                self.solution_matrix[i + 1][-1] = a_vec[-1]

            elif self.bound_cond == 'neumann':
                a_vec[0] = -self.p_func(i)
                a_vec[-1] = self.q_func(i)
                # a_vec[0]= -a_vec[0]
                u_j1 = np.linalg.solve(self.a, np.dot(self.b, self.solution_matrix[i]) + 2 * self.deltax * self.lmbda * a_vec)
                # setting boundary conditions
                self.solution_matrix[i + 1] = u_j1[:]

            elif self.bound_cond == 'periodic':
                self.solution_matrix[i + 1] = np.linalg.solve(self.a, np.dot(self.b, self.solution_matrix[i]))

            elif self.bound_cond == 'homogenous':
                self.solution_matrix[i + 1][1:-1] = np.linalg.solve(self.a,
                                                                    np.dot(self.b, self.solution_matrix[i][1:-1]))
        return self.solution_matrix


def p(t):
    return 5


def q(t):
    return 3


def solve_mypde(method, bound_cond,K,L,T,title):
    # Set problem parameters/functions


    # Set numerical parameters
    mx = 10  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time`

    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = K * deltat / (deltax ** 2)  # mesh fourier number
    print("deltax=", deltax)
    print("deltat=", deltat)
    print("lambda=", lmbda)

    # Set up the solution variables
    u_j = np.sin(pi * x / L).T  # u at current time step

    solver = method(u_j, mx, lmbda, mt, p, q, bound_cond, deltax)
    solver.solve()

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-K * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    pl.plot(x,solver.solution_matrix[-1, :], 'ro', label='num')
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.title(title)
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')
    pl.grid()
    pl.show()



    return


if __name__ == '__main__':
    solver = crankNicholson
    K, L, T = 5,1,0.5

    solve_mypde(solver, 'homogenous',K,L,T,'Crank Nickolson')
    solve_mypde(forwardEuler, 'homogenous', K, L, T,'Forward Euler')
    solve_mypde(backwardEuler, 'homogenous', K, L, T,'Backward Euler')


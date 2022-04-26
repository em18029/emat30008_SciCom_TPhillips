import math
import numpy as np
import matplotlib.pyplot as plt
import time
from odes import odes


def euler_step(tn, xn, step_size, myode, args):
    """
    A function which completes one iteration of the Euler method

    Parameters
    ----------
    tn: float
        The current value of t

    xn: list
        Current position values

    step_size: numpy.float64
        The step size of the iteration

    myode: function
        ode to solve

    args: numpy.float64
        A tuple of float values for parameters of the ODE

    Returns
    -------
    xn+1: float
        Value of x at next timestep
    """
    return xn + step_size * myode(tn, xn, *args)


def RK4(tn, xn, step_size, myode, args):
    """
        A function which completes one iteration of the 4th order Runge Kutta method

        Parameters
        ----------
        tn: float
            The current value of t

        xn: list
            Current position values

        step_size: numpy.float64
            The step size of the iteration

        myode: function
            ode to solve

        args: numpy.float64
            A tuple of float values for parameters of the ODE

        Returns
        -------
        xn+1: float
            Value of x at next timestep
        """

    a = myode(tn, xn, *args)
    b = myode(tn + step_size / 2, xn + a * (step_size / 2), *args)
    c = myode(tn + step_size / 2, xn + a * (step_size / 2), *args)
    d = myode(tn + step_size, xn + c * step_size, *args)
    return xn + ((a + 2 * b + 2 * c + d) / 6) * step_size


def solve_to(x0, t0, t_end, deltat, solver, myode, args):
    """
        A function which solves an ODE with a specified method between points in time

        Parameters
        ----------
        x0: float
            The current value of x
        t0: float
            Initial time value
        tn: float
            Final time value
        deltat: numpy.float64
            Change in time
        solver: function
            Method of solving ode, either Euler or RK
        myode: function
            ODE to solve
        args: numpy.float64
            A tuple of float values for parameters of the ODE

        Returns
        -------
        xn+1: float
            Value of x at next timestep
        """

    x, t = x0, t0
    while t < t_end:
        x = solver(t, x, deltat, myode, args)
        t += deltat
    return x


def solve_myode(t_vals, x0, deltat, solver, myode, args):
    """
            A function which solves an ODE with a specified method between points in time

            Parameters
            ----------
            x0: float
                The current value of x
            t0: float
                Initial time value
            tn: float
                Final time value
            deltat: numpy.float64
                Change in time
            solver: function
                Method of solving ode, either Euler or RK4
            myode: function
                ODE to solve
            args: numpy.float64
                A tuple of float values for parameters of the ODE

            Returns
            -------
            x_vals: list
                A list of solution values
            """


    x_vals = [None] * len(t_vals)
    x_vals[0] = x0
    for i in range(len(t_vals) - 1):
        xn, tn, tn1 = x_vals[i], t_vals[i], t_vals[i + 1]
        x_vals[i + 1] = solve_to(xn, tn, tn1, deltat, solver, myode, args)
    return x_vals


def plot_error(t_vals, x0, myode, exact_vals, args):
    x_val = exact_vals(t_vals[1], 0)
    t_steps = np.logspace(-6, 0, 10)  # logarithmic scale for time
    eul_err = np.zeros(len(t_steps))
    RK4_err = np.zeros(len(t_steps))
    error_match = 1e-5
    eul_t = 0
    RK4_t = 0
    for i in range(len(t_steps)):
        eul_pred = solve_myode(t_vals, x0, t_steps[i], euler_step, myode, args)
        RK4_pred = solve_myode(t_vals, x0, t_steps[i], RK4, myode, args)

        eul_err[i] = abs(eul_pred[-1] - x_val)
        RK4_err[i] = abs(RK4_err[-1] - x_val)

    plt.loglog(t_steps, eul_err, label='Euler')
    plt.loglog(t_steps, RK4_err, label='RK4')
    plt.legend()
    plt.ylabel("Error of approximation")
    plt.xlabel("Timestep")
    plt.grid()
    plt.show()
    return eul_err, RK4_err


# def plot_dot_x(x_vals, t_vals, step_size, myode, myode_exact, args):
#     eul_vals = np.array(solve_myode(t_vals, x_vals, step_size, euler_step, myode, args))
#     RK4_vals = np.array(solve_myode(t_vals, x_vals, step_size, RK4, myode, args))
#     vals_true = [0] * len(t_vals)
#     for i in range(len(vals_true)):
#         vals_true[i] = myode_exact(t_vals[i], x_vals, *args)
#     plt.plot(eul_vals[:, 0], eul_vals[:, 1])
#     plt.plot(RK4_vals[:, 0], RK4_vals[:, 1])
#     plt.grid()
#     plt.show()


if __name__ == "__main__":
    t_vals = [0, 1]
    args = []
    eul_err, RK4_err = plot_error(t_vals,1,odes.ode_second_order,odes.exact_second_order, args)
    t_vals = np.linspace(t_vals[0], t_vals[1], 20)
    x_vals = np.array([3, 4])
    #plot_dot_x(x_vals, t_vals, 0.1, odes.ode_second_order, odes.exact_second_order, args)

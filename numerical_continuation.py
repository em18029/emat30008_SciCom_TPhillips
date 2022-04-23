import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import odes
from shooting import numericalShooting


def natural_continuation(max_steps, par0, vary_par, myode, step_size, u0):
    n = 0
    vary_par_h = []
    sol = []
    while n < max_steps:
        n = n + 1
        par0[vary_par] = par0[vary_par] + step_size
        u0 = numericalShooting.main_loop(myode, u0, par0)
        # print(par0)
        # print(u0)
        # print(' ')
        # save solution and alpha
        sol = np.concatenate((sol,u0),axis=0)
        vary_par_h.append(par0[vary_par])
    return sol, vary_par_h


def pseudo_continuation():
    return


def plot_sols(sol,par_hist):
    plt.subplot(1, 2, 1)
    x_data = sol[:,0]
    y_data = sol[:,1]
    plt.plot(par_hist, x_data, 'r-', label='a')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Varied Parameter')
    plt.ylabel('x')
    plt.subplot(1, 2, 2)
    plt.plot(par_hist, y_data, 'b-', label='b')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Varied Parameter')
    plt.ylabel('x')
    plt.show()

def continuation(
        myode,  # the ODE to use
        u0,  # the initial state
        par0,  # the initial parameters (args)
        vary_par,  # the parameter to vary
        step_size,  # the size of the steps to take
        max_steps,  # the number of steps to take
        discretisation,  # the discretisation to use
        solver,  # the solver to use
        tol  # tolerance for difference between converging parameters
):
    if discretisation == "natural":
        sol, par_hist = natural_continuation(max_steps, par0, vary_par, myode, step_size, u0)
        sol = sol.reshape(max_steps, 3)
        print(sol)
        print(par_hist)

    if discretisation == "pseudo":
        sol, par_hist = pseudo_continuation()

    plot_sols(sol,par_hist)


continuation(
    myode=odes.hopf_bifurcation,  # the ODE to use
    u0=[1.5, 0, 20],  # the initial state
    par0=[0, -1],  # the initial parameters (args)
    vary_par=0,  # the parameter to vary
    step_size=0.2,  # the size of the steps to take
    max_steps=10,  # the final varying parameter value
    discretisation="natural",  # the discretisation to use
    solver=None,  # the solver to use
    tol=1e-6
)

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import odes
from shooting import numericalShooting as shooting


def shooting_continuation(par0, vary_par, par_range, u0, myode, tol):
    for i in par_range:
        par0[vary_par] = i
        u01 = shooting.main(myode, u0, par0)
        if np.allclose(u01, u0, atol = tol):
            print('Parameters converged :',u01)
            return u0
        u0 = u01
    return u0


def notdiscretisation_continuation(discretisation, solver, u0, par0, par_range, vary_par, tol):
    for i in par_range:
        par0[vary_par] = i
        u01 = solver(discretisation, u0)
        if np.allclose(u01, u0, atol = tol):
            print('Parameters converged :',u01)
            return u0
        u0 = u01
    return u0


def numerical_continuation(
        myode,  # the ODE to use
        u0,  # the initial state
        par0,  # the initial parameters (args)
        vary_par,  # the parameter to vary
        step_size,  # the size of the steps to take
        max_steps,  # the number of steps to take
        var_par_end,  # the final varying parameter value
        discretisation,  # the discretisation to use
        solver,  # the solver to use
        tol  # tolerance for difference between converging parameters
):
    # Range for parameter to be varied upon
    par_range = np.arange(par0[vary_par], var_par_end + step_size, step_size)

    if discretisation == shooting:
        u0 = shooting_continuation(par0, vary_par, par_range, u0, myode, tol)
    else:
        u0 = notdiscretisation_continuation(discretisation, solver, u0, par0, par_range, vary_par, tol)
    return


numerical_continuation(
    myode=odes.hopf_bifurcation,  # the ODE to use
    u0=[1.5, 0, 20],  # the initial state
    par0=[1, 0],  # the initial parameters (args)
    vary_par=1,  # the parameter to vary
    step_size=0.5,  # the size of the steps to take
    max_steps=None,  # the number of steps to take
    var_par_end=2,  # the final varying parameter value
    discretisation=shooting,  # the discretisation to use
    solver=scipy.optimize.fsolve,  # the solver to use
    tol=1e-6
)

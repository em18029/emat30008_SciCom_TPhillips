import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def ode_num(t, x_vals, a, b, d):
    x = x_vals[0]
    y = x_vals[1]
    x_array = np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])
    return x_array


def isolate_orbit(ode_data, t_data):
    x_data = ode_data[:, 0]
    y_data = ode_data[:, 1]
    maximums = argrelextrema(x_data, np.greater)[0]
    prev_val = False
    prev_t = 0
    for i in maximums:
        if prev_val:
            if math.isclose(x_data[i], prev_val, abs_tol=1e-4):
                period = t_data[i] - prev_t
                return x_data[i], y_data[i], period
        prev_val = x_data[i]
        prev_t = t_data[i]
    return


def shooting_conditions(ode, u0, args):
    x0 = u0[:-1]
    t0 = u0[-1]
    sol = solve_ivp(ode, (0, t0), x0, max_step=1e-2, args=args)
    x_conds = x0 - sol.y[:, -1]
    t_conds = np.asarray(ode(t0, x0, *args)[0])
    g_conds = np.concatenate((x_conds, t_conds), axis=None)
    return g_conds


def shooting(ode, u0, args):
    """"
    A function that uses numerical shooting to find limit cycles of a specified ODE.

    Parameters
    ode : function
        The ODE to apply shooting to. The ode function should take arguments for the independent variable, dependant
        variable and constants, and return the right-hand side of the ODE as a numpy.array.
    u0 : numpy.array
        An initial guess at the initial values for the limit cycle.

    Returns
    Returns a numpy.array containing the corrected initial values
    for the limit cycle. If the numerical root finder failed, the
    returned array is empty.
    """
    final = fsolve(lambda x: shooting_conditions(ode, x, args=args), u0)
    return final


args = np.array([1, 0.2, 0.1])
print(shooting(ode_num, np.array([1, 1, 20]), args))

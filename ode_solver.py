import math
import numpy as np
import matplotlib.pyplot as plt
import time
from odes import odes


def euler_step(tn, xn, step_size, ode, args):
    return xn + step_size * ode(tn, xn, *args)


def RK4(tn, xn, step_size, ode, args):
    a = ode(tn, xn, *args)
    b = ode(tn + step_size / 2, xn + a * (step_size / 2), *args)
    c = ode(tn + step_size / 2, xn + a * (step_size / 2), *args)
    d = ode(tn + step_size, xn + c * step_size, *args)
    return xn + ((a + 2 * b + 2 * c + d) / 6) * step_size


def solve_to(x0,  t0, t_end,deltat, solver, ode, args):
    x, t = x0, t0
    while t < t_end:
        x = solver(t, x, deltat, ode, args)
        t += deltat
    return x


def solve_ode(t_vals, x0, deltat, solver, ode, args):
    x_vals = [0] * len(t_vals)
    x_vals[0] = x0
    for i in range(len(t_vals) - 1):
        xn, tn, tn1 = x_vals[i], t_vals[i], t_vals[i+1]
        x_vals[i + 1] = solve_to(xn, tn, tn1, deltat, solver, ode, args)
    return x_vals

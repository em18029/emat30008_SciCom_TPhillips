import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import odes
from shooting import numericalShooting



myode= odes.pred_prey,  # the ODE to use
u0 = [1, 0.5, 100]  # the initial state
par0 = np.array([1, 0.1, 0.1])  # the initial parameters
vary_par = 1  # the parameter to vary
step_size = 0.05   # the size of the steps to take
max_steps = 1   # the number of steps to take
var_par_end = 0.4  # the final varying parameter value
discretisation = lambda x : x**3 -x + par0[vary_par]   # the discretisation to use
solver = scipy.optimize.fsolve  # the solver to use


par_range = np.arange(par0[vary_par],var_par_end+step_size,step_size)
args = np.array([1, 0.2, 0.1])
u0 = [1, 0.5, 100]

if discretisation is shooting:
    for i in par_range:
        par0[vary_par] = i
        u0 = numericalShooting.main(odes.pred_prey, u0, par0)
        print(u0)

# elif discretisation is 'not needed':
#     for i in par_range:
#         fsolve()

else:







#from ode_solver import solve_ode
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Global vars
a = 1
d = 0.1


def solve_ode(f,inital_conditions,timesteps):
    return odeint(f,inital_conditions,timesteps)

def f(X,t):
    x, y = X
    dxdt = x*(1-x)- a*x*y/(d+x)
    dvdt = b*y*(1-(y/x))
    return [dxdt, dvdt]

def periodic_orbit():

    return

initial_conditions = [1,1]
timesteps = np.linspace(0, 200, 200)

for b in np.arange(0,1,0.2):
    dXdt = solve_ode(f,initial_conditions,timesteps)
    dxdt = dXdt[:,1]
    dydt = dXdt[:,1]
    print(dxdt)
    print(dydt)

    # plot
    plt.figure()
    plt.plot(timesteps, dXdt[:, 0], label='x');
    plt.plot(timesteps, dXdt[:, 1], label='y');

    plt.legend();
    plt.xlabel('time');
    plt.show()









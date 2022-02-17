import numpy as np
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.optimize import root
import matplotlib.pyplot as plt
from math import nan
import random
# %%


def eqn(X,t,a,d,b):
    x,y = X
    xdot=x*(1-x)- a*x*y/(d+x)
    ydot = b*y*(1-(y/x))
    return np.array([xdot,ydot])

def main_LV_eq():
    #Initial conditions
    a=1
    d=0.1
    # if b<0.26 population of predator and prey fluctuates, of b>0.26 population stabailisegit
    b=0.4
    x0=4
    y0=4
    #Time line setting
    Nt=1000
    t_max=1000
    t=np.linspace(0,t_max,Nt)
    X0=[x0,y0]


    res = integrate.odeint(eqn, X0, t, args = (a, d, b))
    x,y=res.T

    plt.figure()
    plt.grid()
    plt.title("odeint method")
    plt.plot(t, x, 'b', label = 'xdot')
    plt.plot(t, y, 'r', label = "ydot")
    plt.xlabel('Time t, [days]')
    plt.ylabel('Population')
    plt.legend()

    plt.show()
    return


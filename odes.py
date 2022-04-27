import numpy as np
import math



def ode_first_order(t, u):
    return u

def exp(t, u):
    return math.exp(t)

def ode_second_order(t, u):
    u_array = np.array([u[1], -u[0]])
    return u_array

def exact_second_order(t, u):
    x,y = u[0],u[1]
    u_array = np.array([x * math.cos(t) + y * math.sin(t), -x * math.sin(t) + y * math.cos(t)])
    return u_array

def pred_prey(t, vals, a, b, d):
    """
    x_array: array containing system of odes
    """
    x, y = vals[0], vals[1]
    x_array = np.array([x * (1 - x) - (a * x * y) / (d + x), b * y * (1 - (y / x))])
    return x_array


def hopf_bifurcation(t, vals, beta):
    """
    x_array: array containing system of odes
    """
    #beta,sigma = args[0],args[1]
    x, y = vals[0], vals[1]
    x_array = np.array([beta * x - y + -x * (x ** 2 + y ** 2), x + beta * y + -y * (x ** 2 + y ** 2)])
    return x_array

def hbf_analytical(vals):
    t, beta = vals[-1],vals[1]
    x_array = np.array([beta ** 0.5 * math.cos(t), beta ** 0.5 * math.sin(t)])
    return x_array

def alg_cubic(t, x, c):
    return x**3 - x + c



import numpy as np

def hopf_birfurcation(t,vals,args):
    """
    x_array: array containing system of odes
    """
    x,y = vals[0],vals[1]
    beta,sigma = args[0],args[1]
    x_array = np.arrray([beta*x,-y+sigma*x(x**2+y**2)],[x+beta*y+sigma*y(x**2+y**2)])
    return x_array

def pred_prey(t, vals, args):
    """
    x_array: array containing system of odes
    """
    a, b, d = args[0],args[1],args[2]
    x = vals[0]
    y = vals[1]
    x_array = np.array([x*(1-x) - (a*x*y)/(d+x), b*y*(1-(y/x))])
    return x_array
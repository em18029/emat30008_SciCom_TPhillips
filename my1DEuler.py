import numpy as np
import pylab as pl
from math import pi
from scipy.optimize import fsolve

'''
simple forward Euler solver for the 1D heat equation
  u_t = kappa u_xx  0<x<L, 0<t<T
with zero-temperature boundary conditions
  u=0 at x=0,L, t>0
and prescribed initial temperature
  u=u_I(x) 0<=x<=L,t=0
'''

# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for

# Set numerical parameters
mx = 30  # number of gridpoints in space
mt = 1000  # number of gridpoints in time`

x = np.linspace(0, L, mx + 1)  # mesh points in space
t = np.linspace(0, T, mt + 1)  # mesh points in time
deltax = x[1] - x[0]  # gridspacing in x
deltat = t[1] - t[0]  # gridspacing in t
lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
print("deltax=", deltax)
print("deltat=", deltat)
print("lambda=", lmbda)

# Set up the solution variables
u_j = np.zeros(x.size)  # u at current time step
u_jp1 = np.zeros(x.size)  # u at next time step


def u_i(x):
    # initial temperature distribution
    return np.sin(pi * x / L)


def u_exact(x, t):
    # the exact solution
    return np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)


def tridiag_mat():
    # A function generating a tridiagonal (m-1) x (m-1) matrix
    return np.eye(mx + 1, mx + 1, k=-1) * lmbda + np.eye(mx + 1, mx + 1) * (1 - 2 * lmbda) + np.eye(mx + 1, mx + 1,k=1) * lmbda



def u_j_mat():
    return np.zeros((mx +1, mx +1))


# Set initial conditions
u_jp = u_j_mat()
u_jp[:, 0] = u_i(x).T
print(u_i(x).T)
for i in range(1, mx + 1):
    u_jp[:, i] = np.matmul(tridiag_mat(), u_jp[:, i - 1])

#set boundary conditions
u_jp[:, 0] = 0
u_jp[:, -mx] = 0
print(u_jp)

pl.plot(x, u_jp, 'ro', label='num')
xx = np.linspace(0, L, 250)
pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()

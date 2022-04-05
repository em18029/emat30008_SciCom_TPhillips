import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
A = 1e13 # Arrhenius constant
T = 293.15 # Temperature [K]
E_a = 80000 # Activation energy [J/mol]
R = 8.31 # Ideal gas constant [J/molK]
rho = 1000 # density [kg/m3]
F_in = 0.2 # Inlet flowrate [m3/s]
h = 2.1 # Height of reactor
A_ = 1 # Cross-sectional area of reactor [m2]

V = A_*h  # Reactor volume [m3]
k = A*np.exp(-E_a/(R*T)) # Reaction rate constant
F_out = F_in # Steady state

def dUdt(U, t):
    m, C_A, C_B = U

    dmdt = rho*(F_in - F_out)
    dC_Adt = ( F_in*C_A - F_out*C_A )/ V-k*C_A
    dC_Bdt = ( F_in*C_B - F_out*C_B )/ V+k*C_A

    return [dmdt, dC_Adt, dC_Bdt]

# Create time domain
t_span = np.linspace(0, 100, 30)

# Initial condition
C_A0 = 3 # Initial concentration [mol/m3]
C_B0 = 0 # Initial concentration [mol/m3]
m0 = 0 # Initial mass in tank [kg]

Uzero = [m0, C_A0, C_B0]

solution = odeint(dUdt, Uzero, t_span)

# plot
plt.plot(t_span, solution[:, 0], label='masse');
plt.plot(t_span, solution[:, 1], label='C_A');
plt.plot(t_span, solution[:, 2], label='C_B');
plt.legend();
plt.xlabel('time');
plt.show()
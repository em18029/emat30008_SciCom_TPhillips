import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.optimize import fsolve
from odes import odes


class numerical_shooting:
    def __init__(self, ode, u0, args):
        self.ode = ode
        self.ode_data = None
        self.t_data = None
        self.u0 = u0
        self.args = args

    def gen_data(self, args, T):
        '''
        inputs: ODE to solve,args,T
        '''
        t_data= (0,T)
        x0 = [1,0.5]
        ode_data = solve_ivp(self.ode, t_data, x0, max_step=1e-2, args=args)

        self.ode_data = np.transpose(ode_data.y)
        self.t_data = np.transpose(ode_data.t)

    def isolate_orbit(self):
        '''
        inputs: ODE data,t_Data
        '''
        x_data = self.ode_data[:, 0]
        y_data = self.ode_data[:, 1]
        # plt.plot(t_data,x_data)
        # plt.show()
        extrema = argrelextrema(x_data, np.greater)[0]
        print(extrema)
        prev_val = False
        prev_t = 0

        for i in extrema:
            if prev_val:
                print('prev val',prev_val)
                #if math.isclose(x_data[i], prev_val, abs_tol=1e-4):
                period = self.t_data[i] - prev_t
                self.u0 = [x_data[i], y_data[i], period]
                return
            prev_val = x_data[i]
            prev_t = self.t_data[i]
        return

    def shooting_conditions(self, u0, ode, args):
        '''
        inputs: ODE,U0,args
        '''
        x0 = u0[:-1]
        t0 = u0[-1]
        sol = solve_ivp(ode, (0, t0), x0, max_step=1e-2, args=args)
        x_conds = x0 - sol.y[:, -1]
        t_conds = np.asarray(ode(t0, x0, *args)[0])
        g_conds = np.concatenate((x_conds, t_conds), axis=None)
        return g_conds


    def shooting(self):
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
        final = fsolve(self.shooting_conditions, self.u0, args=(self.ode, self.args))
        return final

args = np.array([1, 0.2, 0.1])
u0 = [1,0.5,100]
nS = numerical_shooting(odes.pred_prey,u0,args)


nS.gen_data(args, 100)
nS.isolate_orbit()

print('Initial starting conditions', nS.u0)
print(nS.shooting())

#ode_data,t_data = gen_data(odes.pred_prey, args, 100)
#x,y,period = isolate_orbit(ode_data, t_data)

# u0=[x,y,period]
# print('Initial starting conditions',u0)
# print(shooting(ode_to_solve, u0, args))




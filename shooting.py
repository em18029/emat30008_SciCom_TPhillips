import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import pred_prey


class numericalShooting:
    def __init__(self, ode, u0, args):
        self.ode = ode
        self.ode_data = None
        self.t_data = None
        self.u0 = u0
        self.args = args

    def gen_data(self, args):
        '''
        A function that solves a specified ODE using fsolve and returns the relevant ODE data and time points.

        Parameters
        ----------
        self: class variable
        args: list of specified parameters relevant to the ODE being solved in the numericalShooting class

        Returns
        -------
        Update to ode_data and t_data in the numericalShooting class
        '''
        x0 = self.u0[:-1]
        T = self.u0[-1]
        t_data = (0, T)
        ode_data = solve_ivp(self.ode, t_data, x0, max_step=1e-2, args=args)
        self.ode_data = np.transpose(ode_data.y)
        self.t_data = np.transpose(ode_data.t)

    def isolate_orbit(self):
        '''
        A function that isolates the orbit of the ODE, finding the period with corresponding x and y points

        Returns
        -------
        An update to u0 where x_data and y_data are boundary values found and and an update on period.
        '''
        x_data = self.ode_data[:, 0]
        y_data = self.ode_data[:, 1]
        # plt.plot(t_data,x_data)
        # plt.show()
        extrema = argrelextrema(x_data, np.greater)[0]
        prev_val = False
        prev_t = 0

        for i in extrema:
            if prev_val:
                # if math.isclose(x_data[i], prev_val, abs_tol=1e-4):
                period = self.t_data[i] - prev_t
                self.u0 = [x_data[i], y_data[i], period]
                return
            prev_val = x_data[i]
            prev_t = self.t_data[i]
        return

    def shooting_conditions(self, u0, ode, args):
        '''
        A function which utilises scipy's solve_ivp to find the starting



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

    def main( ode, u0, args):
        nS = numericalShooting(ode, u0, args)
        nS.gen_data(args)
        nS.isolate_orbit()
        print('Initial starting conditions', nS.u0)
        print('Initial limit conditions', nS.shooting())

        return nS.shooting()


if __name__ == '__main__':
    '''
    Predator Prey equations init conditions
    '''
    args = np.array([1, 0.2, 0.1])
    u0 = [1, 0.5, 100]
    nS = numericalShooting(pred_prey, u0, args)

    nS.gen_data(args)
    nS.isolate_orbit()
    print('Initial starting conditions', nS.u0)
    print('Initial limit conditions', nS.shooting())

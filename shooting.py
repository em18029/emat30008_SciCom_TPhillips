import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import pred_prey, hopf_bifurcation
from ode_solver import solve_myode,RK4

'''
Add plotting one limit cycle and other full data from odes
Add checking inputs
try get isclsoe to work
'''


class numericalShooting:
    '''
    A class that applies a numerical shooting method to a system of ODEs.

    Parameters
    ----------
    self: class variable
    ode:  function
        the n dimensional system of odes to be solved
    ode_data:  numpy.array
        data drawn from the solved ODEs
    t_data:  numpy.array
        corresponding time data to ode_data
    u0:  numpy.array
        initial guess of starting conditions
    args:  list
        parameters of the system of ODEs
    '''
    def __init__(self, ode, u0, args):
        self.ode = ode
        self.ode_data = None
        self.t_data = None
        self.u0 = u0
        self.args = args
        self.max_step = 1e-2
        self.x_data = None
        self.y_data = None
        self.final = None

    def gen_data(self, args):
        '''
        A function that solves a specified ODE using solve_ivp and returns the relevant ODE data and time points.

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
        ode_data = solve_ivp(self.ode, t_data, x0, max_step=self.max_step, args=args)
        self.ode_data = np.transpose(ode_data.y)
        self.t_data = np.transpose(ode_data.t)
        return self.ode_data, self.t_data

    def isolate_orbit(self):
        '''
        A function that isolates the orbit of the ODE, finding the period with corresponding x and y points

        Returns
        -------
        An update to u0 where x_data and y_data are boundary values found and and an update on period.
        '''
        self.x_data, self.y_data = self.ode_data[:, 0], self.ode_data[:, -1]
        prev_val = False
        prev_t = 0
        self.extrema = find_peaks(self.x_data)[0]
        for i in self.extrema:
            if prev_val:
                if math.isclose(self.x_data[i], prev_val, abs_tol=0.5):
                    period = self.t_data[i] - prev_t
                    self.u0 = [self.x_data[i], self.y_data[i], period]
                return self.u0
            prev_val = self.x_data[i]
            prev_t = self.t_data[i]
        #raise RuntimeError("No orbit found")

    def shooting_conditions(self, u0, ode, args):
        '''
        A function which utilises scipy's solve_ivp to find the best guess for solve_ivp
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
        ----------
        ode  function
            The ODE to apply shooting to. The ode function should take arguments for the independent variable, dependant
            variable and constants, and return the right-hand side of the ODE as a numpy.array.
        u0  numpy.array
            An initial guess at the initial values for the limit cycle.

        Returns
        -------
        Returns a numpy.array containing the corrected initial values
        for the limit cycle. If the numerical root finder failed, the
        returned array is empty.
        """
        final = fsolve(self.shooting_conditions, self.u0, args=(self.ode, self.args))
        self.final = final
        return final

    def main_loop(ode, u0, args):
        '''
        A function that replaces if __name__ == '__main__', such that the numericalShooting class can be ran from
        external script

        Parameters
        ----------
        u0:  numpy.array
            initial guess of starting conditions
        args:  list
            parameters of the system of ODEs

        Returns
        -------
        Initial limit conditions for the system of ODEs containing, if the numerical shooting method failed then the
        output willl be empty.
        '''
        nS = numericalShooting(ode, u0, args)
        nS.gen_data(args)
        nS.isolate_orbit()
        # print('Initial starting conditions', nS.u0)
        # print('Initial limit conditions', nS.shooting())
        final = nS.shooting()
        # nS.plot_sols(args)
        return final

    def plot_sols(self, args):
        plt.subplot(1, 2, 1)
        plt.plot(self.t_data, self.x_data, 'r-', label='x')
        plt.plot(self.t_data, self.y_data, 'b-', label='y')
        plt.grid()
        plt.title('hopf-Bifurcation solutions')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.subplot(1, 2, 2)
        self.u0 = self.final
        # print('u0', self.u0)
        nS.gen_data(args)
        plt.title('One limit Cylce of hopf-Bifurcation')
        plt.plot(self.t_data, self.ode_data[:, 0], 'r-', label='x')
        plt.plot(self.t_data, self.ode_data[:, 1], 'b-', label='y')
        plt.plot([0, self.final[-1]], [self.final[0], self.final[0]], 'ro')
        plt.plot([0, self.final[-1]], [self.final[1], self.final[1]], 'bo')
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.show()


if __name__ == '__main__':
    '''
    Predator Prey equations init conditions
    '''
    # args = np.array([1, 0.25, 0.1])
    # u0 = [5, 10, 200]
    # myode = pred_prey
    #
    '''
    hopf-Bifurcation equtions
    '''
    args = np.array([0])
    u0 = [0.5, 0.5, 40]
    myode = hopf_bifurcation

    # args = np.array([0])
    # u0 = [3.16227766e-01, 1.82691380e-11, 6.28318531e+00]
    # myode = hopf_bifurcation

    nS = numericalShooting(myode, u0, args)
    nS.gen_data(args)
    nS.isolate_orbit()
    print('Initial starting conditions', nS.u0)
    print('Initial limit conditions', nS.shooting())

    nS.plot_sols(args)





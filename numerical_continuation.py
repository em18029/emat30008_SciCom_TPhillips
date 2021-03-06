import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import pred_prey, hopf_bifurcation, alg_cubic
from shooting import numericalShooting
from ode_solver import solve_myode, RK4
from pde_solver import solve_mypde


def natural_continuation(p, par, vary_par, myode, step_size, u0, limit_cycle):
    """
        A function which performs natural parameter continuatin of a set of ODEs

        Parameters
        ----------
        p  A list of min and max varying parameter values
        par  The parameters of the odes
        vary_par  The index of the parameter to be varied
        myode  An arbitart system of odes to be solved
        step_size  The step size for the varying parameter to be updated by
        u0  Initial condidions
        limit_cycle  a boolean, true or false

        Returns
        -------
        sol numpy.array
            a list of soltions calcualted from either numerical shooting for fsolve
        vary_par_h numpy.array
            a list of all previous varied parameter values
        """

    n = 0
    vary_par_h = []
    sol = []
    while par[vary_par] < p[1]:
        n = n + 1
        par[vary_par] = par[vary_par] + step_size
        if limit_cycle:
            u0 = numericalShooting.main_loop(myode, u0, par)
        else:
            u0 = fsolve(lambda x: myode(0, x, par[vary_par]), np.array(u0))
        sol = np.concatenate((sol, u0), axis=0)
        vary_par_h.append(par[vary_par])

    if limit_cycle:
        return sol.reshape(n, len(u0)), vary_par_h
    else:
        return sol, vary_par_h


def calc_pseudo(state_sec, par_sec, pred_state, pred_par, u, p):
    """
        A function which calcualtes the next parameter value.

        Parameters
        ----------
        state_sec  u0 secant
        par_sec  parameter secant
        pred_state  predicted next state
        pred_par  predicted parameter value
        u  the current state
        p  the current parameter value

        Returns
        -------
        pseudo  float
            next parameter value
        """
    return np.dot(u - pred_state, state_sec) + np.dot(p - pred_par, par_sec)


def pseudo_conds(myode, u0, state_sec, par_sec, pred_state, pred_par, limit_cycle, args):
    """
        A function which outputs time points and values of solution at t

        Parameters
        ----------
        myode  An arbitart system of odes to be solved
        u0  Initial condidions
        state_sec  u0 secant
        par_sec  parameter secant
        pred_state  predicted next state
        pred_par  predicted parameter value
        limit_cycle  a boolean, true or false
        args  The parameters of the odes

        Returns
        -------
        g_conds numpy.array
            a list of solutions in a single state calcualted from solveivp
        """
    if limit_cycle:
        pseudo = calc_pseudo(state_sec, par_sec, pred_state, pred_par, u0[:-1], u0[-1])
        sol = solve_ivp(myode, (0, u0[-2]), u0[:-2], max_step=1e-2, args=args)
        x_conds = u0[:-2] - sol.y[:, -1]
        t_conds = np.asarray(myode(u0[-2], u0[:-2], *args)[0])
        g_conds = np.concatenate((x_conds, t_conds, pseudo), axis=None)
    else:
        x_conds = np.asarray(myode(0, u0[:-1], *args)[0])
        pseudo = calc_pseudo(state_sec, par_sec, pred_state, pred_par, u0[:-1], u0[-1])
        g_conds = np.concatenate((x_conds, pseudo), axis=None)
    return g_conds


def pseudo_continuation(p, par, vary_par, myode, step_size, u0, limit_cycle):
    """
        A function which performs pseudo arclength continuation

        Parameters
        ----------
        p  A list of min and max varying parameter values
        par  The parameters of the odes
        vary_par  The index of the parameter to be varied
        myode  An arbitart system of odes to be solved
        step_size  The step size for the varying parameter to be updated by
        u0  Initial condidions
        limit_cycle  a boolean, true or false

        Returns
        -------
        sol numpy.array
            a list of solutions and varied parameter values
        """
    # p: start finish vals
    # par: initial vals
    # vary_par: par vary index
    n = 0
    if p[1] < p[0]:
        step_size = -step_size

    if limit_cycle:
        # Calc u0,par0
        par0 = list(par)
        par0[vary_par] = par0[vary_par] + step_size
        u0 = numericalShooting.main_loop(myode, u0, par0)
        sol = np.append(u0, par0[vary_par])

        # Calc u1, par1
        par1 = list(par0)
        par1[vary_par] = par1[vary_par] + step_size
        u1 = numericalShooting.main_loop(myode, u0, par0)
        A = np.append(u1, par1[vary_par])

        # Solution array generating
        sol = np.vstack((np.array(sol), np.array(A)))


        # calculate secants
        # -----------------------
        p1 = par1[vary_par]
        p0 = par0[vary_par]
    else:
        par0 = list(par)
        u = list(u0)
        par0[vary_par] += step_size
        u0 = fsolve(lambda x: myode(0, x, par[vary_par]), np.array(u0))
        sol = np.append(u0, par0[vary_par])

        par1 = list(par0)
        par1[vary_par] += step_size
        u1 = fsolve(lambda x: myode(0, x, par[vary_par]), np.array(u))
        A = np.append(u1, par1[vary_par])

        p1 = par1[vary_par]
        p0 = par0[vary_par]

        sol = np.vstack((sol, np.array(A)))
    state_sec = np.array(u1) - np.array(u0)
    par_sec = p1 - p0

    # Predicting states
    pred_state = u1 + state_sec
    pred_par = p1 + par_sec
    par[vary_par] = p1

    if p[1] > p[0]:
        conditions = p[1] > par[vary_par] > p[0]
    else:
        conditions = p[1] < par[vary_par] < p[0]

    while conditions:
        p0, u0 = p1, list(u1)  # Setting up for next time step
        input = np.append(u1, par)
        A = np.array(fsolve(lambda x: pseudo_conds(myode, x, state_sec, par_sec, pred_state, pred_par, limit_cycle, args=par),
                   input))
        sol = np.vstack((sol, A))
        u1, p1 = A[:-1], A[-1]
        # Calculating secants
        state_sec = np.array(u1) - np.array(u0)
        par_sec = p1 - p0

        # Predicting states
        pred_state = u1 + state_sec
        pred_par = p1 + par_sec
        par[vary_par] = p1
        # if sol[-1, -1] > p[1] or sol[-1, -1] < p[0]:
        #     sol = sol[:-1, :]

        if p[1] > p[0]:
            conditions = p[1] > par[vary_par] > p[0]
        else:
            conditions = p[1] < par[vary_par] < p[0]
    return sol


def plot_sols(sol, par_hist):
    par_hist = list(par_hist)
    plt.subplot(1, 2, 1)
    x_data = list(sol[:, 0])
    print(x_data)
    y_data = list(sol[:, 1])
    print(y_data)
    plt.plot(par_hist, x_data, 'r-', label='a')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Varied Parameter')
    plt.ylabel('x')
    plt.subplot(1, 2, 2)
    plt.plot(par_hist, y_data, 'b-', label='b')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Varied Parameter')
    plt.ylabel('x')
    plt.show()

    plt.plot(par_hist, x_data, 'r-', label='x')
    plt.plot(par_hist, y_data, 'b-', label='y')
    plt.ylabel('x')
    plt.xlabel('Varied Parameter')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def continuation(
        myode,  # the ODE to use
        u0,  # the initial state
        par0,  # the initial parameters (args)
        vary_par,  # the parameter to vary
        step_size,  # the size of the steps to take
        p,  # the initial and final value
        discretisation,  # the discretisation to use
        solver,  # the solver to use
        limit_cycle,  # a boolean
        method,  # method for solving ODE can be either euler, RK4, forwardEuler, backwardEuler, crankNicholson
        bound_cond,  # PDE boundary condition
        K,  # diffusion constant
        L,  # length of spatial domain
        T  # total time to solve for
):
    if discretisation == "natural continuation":
        print('Please be patient, it is running!')
        sol, par_hist = natural_continuation(p, par0, vary_par, myode, step_size, u0, limit_cycle)
        print(sol)
        return sol, par_hist
        # if sol.size>1:
        #     if len(sol[0, :]) > 2:
        #         plot_sols(sol, par_hist)
        #         print(sol)
        #     return sol

    elif discretisation == 'pseudo arclength':
        print('Please be patient, it is running!')
        sol = np.array(pseudo_continuation(p, par0, vary_par, myode, step_size, u0, limit_cycle))
        # print(sol)
        if len(sol[0, :-1]) > 2:
            plot_sols(sol[:, :-1], sol[:, -1])
        else:
            plt.plot(sol[:, 1], sol[:, 0])
        # plt.show()
        return sol

    elif discretisation == 'pde solver':
        if bound_cond is 'homogenous' or 'dirichlet' or 'neumann' or 'periodic':
            solve_mypde(method, bound_cond)
            return
        else:
            raise RuntimeError("Invalid boundary condition")

    elif discretisation == 'shooting':
        nS = numericalShooting.main_loop(myode, u0, par0)
        return nS

    elif discretisation == 'ode solver':
        times = np.linspace(0, T, num=600)
        u_values = np.asarray(solve_myode(times, u0, step_size, method, myode, par0))
        return u_values
    else:
        raise RuntimeError(
            "Invalid choice of discretisation, please choose from either: ode solver, shooting, pde solver, natural continuation or pseudo arclength")

    return


if __name__ == "__main__":
    sol = continuation(
        myode=hopf_bifurcation,  # the ODE to use
        u0=[1.2,1.2,20],  # the initial state
        par0=[2],  # the initial parameters (args)
        vary_par=0,  # the parameter to vary
        step_size=0.05,  # the size of the steps to take
        p=[2,0],  # the start and final value of p
        discretisation="pseudo arclength",  # the discretisation to use
        solver=True,  # the solver to use
        limit_cycle=True,  # a boolean
        method=None,   # method for solving ODE can be either euler, RK4, forwardEuler, backwardEuler, crankNicholson
        bound_cond=None,  # PDE boundary condition
        K=None,   # diffusion constant
        L=None,  # length of spatial domain
        T=None   # total time to solve for
    )


    '''
    sol_nat,par_hist = continuation(
        myode=alg_cubic,  # the ODE to use
        u0=[1],  # the initial state
        par0=[-2],  # the initial parameters (args)
        vary_par=0,  # the parameter to vary
        step_size=0.1,  # the size of the steps to take
        p=[-2, 2],  # the start and final value of p
        discretisation="natural continuation",  # the discretisation to use
        solver=True,  # the solver to use
        tol=1e-6,
        limit_cycle=False,
        method=None,
        bound_cond = None,
        K= None,
        L = None,
        T=None
    )
    '''



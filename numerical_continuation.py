import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import pred_prey, hopf_bifurcation, alg_cubic
from shooting import numericalShooting
from ode_solver import solve_myode, RK4


def natural_continuation(p, par, vary_par, myode, step_size, u0, limit_cycle):
    n = 0
    vary_par_h = []
    sol = []
    while par[vary_par] < p[1]:
        n = n + 1
        par[vary_par] = par[vary_par] + step_size
        if limit_cycle:
            u0 = numericalShooting.main_loop(myode, u0, par)
            sol = np.concatenate((sol, u0), axis=0)
            vary_par_h.append(par[vary_par])

        else:
            u0 = fsolve(lambda x: myode(0, x, par[vary_par]), np.array(u0))
            sol = np.concatenate((sol, u0), axis=0)
            vary_par_h.append(par[vary_par])

    if limit_cycle:
        return sol.reshape(n, len(u0)), vary_par_h
    else:
        return sol, vary_par_h


def calc_secants(u0, p0, u1, p1):
    state_sec = np.array(u1) - np.array(u0)
    par_sec = p1 - p0
    pred_state = u1 + state_sec
    pred_par = p1 + par_sec
    return state_sec, par_sec, pred_state, pred_par


def calc_pseudo(state_sec, par_sec, pred_state, pred_par, u, p):
    return np.dot((u - pred_state), state_sec) + np.dot((p - pred_par), par_sec)


def pseudo_conds(myode, u0, state_sec, par_sec, pred_state, pred_par, args):
    pseudo = calc_pseudo(state_sec, par_sec, pred_state, pred_par, u0[:-1], u0[-1])
    sol = solve_ivp(myode, (0, u0[-2]), u0[:-2], max_step=1e-2, args=args)
    x_conds = u0[:-2] - sol.y[:, -1]
    t_conds = np.asarray(myode(u0[-2], u0[:-2], *args)[0])
    g_conds = np.concatenate((x_conds, t_conds, pseudo), axis=None)
    return g_conds


def pseudo_continuation(p, par, vary_par, myode, step_size, u0, limit_cycle):
    # p: start finish vals
    # par: initial vals
    # vary_par: par vary index
    n = 0

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
    sol = np.vstack((sol, A))

    # calculate secants
    print('Calculating secants')
    # -----------------------
    p1 = par1[vary_par]
    p0 = par0[vary_par]
    state_sec = np.array(u1) - np.array(u0)
    par_sec = p1 - p0

    # Predicting states
    pred_state = u1 + state_sec
    pred_par = p1 + par_sec
    par[vary_par] = p1

    # -----------------------

    # begin while loop
    while par[vary_par] < p[1]:
        input = np.append(u1, par)
        A = fsolve(lambda x: pseudo_conds(myode, x, state_sec, par_sec, pred_state, pred_par, args=par), input)
        sol = np.vstack((sol, A))
        print(A)
        p0, u0, u1, p1 = p1, u1, A[:-1], A[-1] # Setting up for next time step

        # Calculating secants

        state_sec = np.array(u1) - np.array(u0)
        par_sec = p1 - p0

        # Predicting states
        pred_state = u1 + state_sec
        pred_par = p1 + par_sec
        par[vary_par] = p1

    return sol


def plot_sols(sol, par_hist):
    plt.subplot(1, 2, 1)
    x_data = sol[:, 0]
    y_data = sol[:, 1]
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


def continuation(
        myode,  # the ODE to use
        u0,  # the initial state
        par0,  # the initial parameters (args)
        vary_par,  # the parameter to vary
        step_size,  # the size of the steps to take
        p,  # the initial and final value
        discretisation,  # the discretisation to use
        solver,  # the solver to use
        tol,  # tolerance for difference between converging parameters
        limit_cycle
):
    if discretisation == "natural":
        print('Please be patient, it is running!')
        sol, par_hist = natural_continuation(p, par0, vary_par, myode, step_size, u0, limit_cycle)
        if len(sol[0, :]) > 2:
            plot_sols(sol, par_hist)
            print(sol)
    elif discretisation == 'pseudo':
        sol = np.array(pseudo_continuation(p, par0, vary_par, myode, step_size, u0, limit_cycle))


continuation(
    myode=hopf_bifurcation,  # the ODE to use
    u0=[1.5, 0, 40],  # the initial state
    par0=[0],  # the initial parameters (args)
    vary_par=0,  # the parameter to vary
    step_size=0.1,  # the size of the steps to take
    p=[0, 2],  # the start and final value of p
    discretisation="pseudo",  # the discretisation to use
    solver=None,  # the solver to use
    tol=1e-6,
    limit_cycle=True
)

# continuation(
#     myode=alg_cubic ,  # the ODE to use
#     u0=np.array([1]),  # the initial state
#     par0=[-2,],  # the initial parameters (args)
#     vary_par=0,  # the parameter to vary
#     step_size=0.2,  # the size of the steps to take
#     p=[-2, 2],  # the start and final value of p
#     discretisation="natural",  # the discretisation to use
#     solver=None,  # the solver to use
#     tol=1e-6,
#     limit_cycle=False
# )

# continuation(alg_cubic, np.array([1]), 0.1, 0, [-2, 2], False, 'natural_parameter', RK4,'n/a', 0, 0, p, q, False, np.array([0], dtype=float))
# (diff_eq, initial_guess, step_size, vary_par_index, vary_range, orbit, discretisation, method,boundary, L, T, p_func, q_func, plot, args):

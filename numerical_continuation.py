import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from odes import pred_prey, hopf_bifurcation, alg_cubic
from shooting import numericalShooting
from ode_solver import solve_myode, RK4


def natural_continuation(p, par0, vary_par, myode, step_size, u0, limit_cycle):
    n = 0
    vary_par_h = []
    sol = []
    while par0[vary_par] < p[1]:
        par0[vary_par] = par0[vary_par] + step_size
        if limit_cycle:
            print(u0)
            print(par0)
            u0 = numericalShooting.main_loop(myode, u0, par0)
            sol = np.concatenate((sol, u0), axis=0)
            vary_par_h.append(par0[vary_par])

        else:
            u0 = fsolve(lambda x: myode(0, x, par0[vary_par]), np.array(u0))
            sol = np.concatenate((sol, u0), axis=0)
            vary_par_h.append(par0[vary_par])

    if limit_cycle: return sol.reshape(n,3), vary_par_h
    else: return sol,vary_par_h




def pseudo_shooting_conds(myode, u0, all_pars, limit_cycle, pseudo):
    par, state_secant, pred_par, par_secant, pred_state = pseudo
    if limit_cycle:
        x0 = u0[:-1]
        t0 = u0[-1]
        times = np.linspace(0, 500, num=2000)
        t_conds = np.array(myode(t0, x0, all_pars)[0])
        #sol = solve_ivp(myode, (0,t0), x0, max_step=1e-2, args=all_pars) # try this
        sol = solve_ivp(myode, times, x0, max_step=1e-2, args=all_pars)
        x_conds = x0 - sol.y[:, -1]
        pseudo = np.dot(u0 - pred_state, state_secant) + np.dot(par - pred_par, par_secant)
        g_conds = np.concatenate((x_conds, t_conds, pseudo), axis=None)
        return g_conds


def pseudo_shooting(myode, u0, all_pars, limit_cycle, pseudo):
    print(u0)
    print(all_pars)
    print(pseudo)
    #return fsolve(lambda x: pseudo_shooting_conds(myode, x, all_pars, limit_cycle, pseudo), u0)
    #return fsolve(lambda x: pseudo_shooting_conds(myode, x, all_pars, limit_cycle, pseudo), u0)
    return fsolve(pseudo_shooting_conds, u0, args=(myode, all_pars))





def pseudo_continuation(p, par, vary_par, myode, step_size, u0, limit_cycle):
    if limit_cycle:
        parn = par
        min_p, max_p = p[0], p[1]
        nS = numericalShooting(myode, u0, par)  # Initialising numerical shooting class
        # vals needed to calculate initial u0
        t_vals = np.linspace(min_p, max_p, int(1000))

        # perform first parameter update, calculating first initial solution
        vals = np.array(solve_myode(t_vals, u0[:-1], step_size, RK4, myode, par))
        nS.ode_data, nS.t_data = vals, t_vals  # Setting values needed for isolate orbit
        nS.isolate_orbit()
        u0 = np.array(nS.u0)
        par[vary_par] += step_size
        par1 = par

        # perform second parameter update, calculating second initial solution
        nS = numericalShooting(myode, u0, par)
        vals = np.array(solve_myode(t_vals, u0[:-1], step_size, RK4, myode, par))
        nS.ode_data, nS.t_data = vals, t_vals  # Setting values needed for isolate orbit
        nS.isolate_orbit()
        u1 = np.array(nS.u0)
        par1[vary_par] += step_size

        # Calculate secants

        state_secant = u1 - u0
        pred_state = u1 + state_secant
        par_secant = par1[vary_par] - par[vary_par]
        pred_par = par1[vary_par] + par_secant
        sols = [np.append(u0, par[vary_par]), np.append(u1, par1[vary_par])]

        n = 0
        while par1[vary_par] < max_p and par1[vary_par] > min_p:
            n += 1
            u0 = u1
            par[vary_par] = par1[vary_par]
            pseudo = [par[vary_par],state_secant, pred_par, par_secant, pred_state]

            if limit_cycle:
                vals = pseudo_shooting(myode, u0, par, limit_cycle, pseudo)
            else:
                vals = fsolve(lambda x: pseudo_shooting(myode, x, pseudo, limit_cycle, args=par1),
                              np.append(u1, par1[vary_par]))


            '''
            Todo: work on update current state, think everything is working up untill that point, check correct vals are 
            outputter
            '''
            # Update current state
            u1 = vals[:-1]
            # Calculate secants
            state_secant = u1 - u0
            pred_state = u1 + state_secant
            par1[vary_par] = vals[-1]
            par_secant = par1[vary_par] - par[vary_par]
            pred_par = par1[vary_par] + par_secant
            sols.append(vals)
        return sols












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
        sol, par_hist = natural_continuation(p, par0, vary_par, myode, step_size, u0,limit_cycle)
        #plot_sols(sol, par_hist)
        print(sol)
    elif discretisation == 'pseudo':
        sol = np.array(pseudo_continuation(p, par0, vary_par, myode, step_size, u0, limit_cycle))
        print('sols', sol)


continuation(
    myode=hopf_bifurcation,  # the ODE to use
    u0=[0.5, 0.5, 20],  # the initial state
    par0=[0, -1],  # the initial parameters (args)
    vary_par=0,  # the parameter to vary
    step_size=0.1,  # the size of the steps to take
    p=[0, 2],  # the start and final value of p
    discretisation="natural",  # the discretisation to use
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

#continuation(alg_cubic, np.array([1]), 0.1, 0, [-2, 2], False, 'natural_parameter', RK4,'n/a', 0, 0, p, q, False, np.array([0], dtype=float))
#(diff_eq, initial_guess, step_size, vary_par_index, vary_range, orbit, discretisation, method,boundary, L, T, p_func, q_func, plot, args):

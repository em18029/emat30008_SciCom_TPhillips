import unittest
import numpy as np
from shooting import numericalShooting
from odes import pred_prey
from scipy.integrate import solve_ivp
import math


class MyTestCase(unittest.TestCase):
    # def test_nullcline(self):
    #     [u1,u2] = nullcline(odes.pred_prey,[0,1])
    #     assert np.allclose([])

    def test_gen_ode_data(self):
        self.assertIsNotNone(nS.ode_data, 'ODE data generated')

    def test_gen_t_data(self):
        self.assertIsNotNone(nS.t_data, 'Time data generated')

    def test_data_shape(self):
        data_shape = np.shape(nS.ode_data)
        self.assertEqual((u0[-1] / nS.max_step + 1, len(args[:-1])), data_shape, 'Data generated is the correct shape')

    def test_extrema(self):
        self.assertIsNotNone(nS.extrema, 'Extrema of odes found')

    # def test_periods_close(self):
    #     self.assertAlmostEqual(nS.find_period(nS.ode_data[:, 0]), nS.find_period(nS.ode_data[:, 0]))

    def test_shooting_conditions(self):
        self.assertIsNotNone(final, 'Shooting conditions outputted')

    def test_hopf_bi_sols(self):
        from odes import hopf_bifurcation_test as func
        from odes import hopf_bifurcation_exact as exact_func
        u0, args = [0.4 ** 0.5, 0, 30], [0.4, -1]

        sol = solve_ivp(func, (0, u0[-1]), u0[:-1], args=args, max_step=1e-2)
        t_conds = sol.t
        x_conds = sol.y

        exact_sol = [[] for i in range(len(x_conds))]
        for t in t_conds:
            sol = exact_func(t, *args)
            for dim in range(len(x_conds)):
                exact_sol[dim].append(sol[dim])

        assert np.allclose(x_conds, np.array(exact_sol))


    def test_hopf_bi_period(self):
        from odes import hopf_bifurcation as func
        t0_exact = 2 * np.pi
        nS = numericalShooting(func,[0.4 ** 0.5, 0, 30],[0.4])

        return np.allclose(t0_exact, nS.u0[-1])



if __name__ == '__main__':
    u0 = np.array([1, 0.5, 100]);
    args = np.array([1, 0.2, 0.1]);
    myode = pred_prey
    nS = numericalShooting(myode, u0, args)
    nS.gen_data(args)
    nS.isolate_orbit()
    final = nS.shooting()

    unittest.main()

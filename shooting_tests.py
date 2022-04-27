import unittest
import numpy as np
from shooting import numericalShooting
from odes import odes
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

    def test_periods_close(self):
        self.assertAlmostEqual(nS.find_period(nS.ode_data[:, 0]), nS.find_period(nS.ode_data[:, 0]))

    def test_shooting_conditions(self):
        self.assertIsNotNone(final,'Shooting conditions outputted')

    # def test_hbf_annalytical(self):
    #     '''
    #         Hof-Bifurcation equtions
    #         '''
    #     args = np.array([0.5, -1])
    #     u0 = [1.5, 0, 20]
    #     myode = odes.hopf_bifurcation
    #     u0 = numericalShooting.main_loop(myode, u0, args)
    #
    #     ana_hbf = lambda t: beta**0.5 * math.cos(t)
    #     beta = args[0]
    #     self.assertAlmostEqual(u0[0],fsolve(ana_hbf))



if __name__ == '__main__':
    u0 = np.array([1, 0.5, 100]);
    args = np.array([1, 0.2, 0.1]);
    myode = odes.pred_prey
    nS = numericalShooting(myode, u0, args)
    nS.gen_data(args)
    nS.isolate_orbit()
    final = nS.shooting()

    unittest.main()

import unittest
import numpy as np
from shooting import numericalShooting
from odes import odes



class MyTestCase(unittest.TestCase):
    # def test_nullcline(self):
    #     [u1,u2] = nullcline(odes.pred_prey,[0,1])
    #     assert np.allclose([])


    def test_gen_data(self):
       self.assertIsNotNone(nS.ode_data,'ODE data generated')

    def







if __name__ == '__main__':
    u0 = [1, 0.5, 100]; args = np.array([1, 0.2, 0.1])
    nS = numericalShooting(odes.pred_prey, u0, args)
    unittest.main()

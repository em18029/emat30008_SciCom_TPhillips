from shooting import numerical_shooting
from ode import hopf_birfurcation
import math

def test(result,known_result):
    if math.close(result,known_result,abs_tol = 1e-4):
        print('Test Passed')
    else:
        print('Test Failed')
    return


#result =

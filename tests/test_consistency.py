# tests/test_consistency.py
import numpy as np
from src.utils import set_seed
from src.montecarlo import mc_standard_estimator, symmetrized_estimator

def linear_f(X):
    # f(x,y,z) = x -> integral over symmetric ball should be 0
    return X[:, 0]

if __name__ == '__main__':
    set_seed(1)
    # Standard MC on linear function (should be ~0)
    est_std, n_std, vals_std = mc_standard_estimator(linear_f, 20000)
    print('MC standard linear function estimate (should be near 0):', est_std)

    # Symmetrized MC should be much closer to zero for linear function
    est_sym, n_sym, _ = symmetrized_estimator(linear_f, 2000)
    print('Symmetrized linear estimate (should be closer to 0):', est_sym)

    # Basic checks
    assert abs(est_sym) < abs(est_std) * 0.5 or abs(est_sym) < 1e-6, \
        "Symmetrized estimator did not reduce the absolute error as expected."
    print("Basic symmetry check passed.")

"""
Monte-Carlo sampling and estimators for integration over the unit ball.

Task 1 (unbiasedness, variance formulas implemented numerically),
Task 3 (rejection sampling), Task 4 (error vs N), Task 5 (symmetrized estimator)

"""
import numpy as np
import itertools
from math import pi
from .utils import f_from_points, volume_ball

def sample_ball_rejection(N, batch_factor=1.5):
    """
    Rejection sampling: propose uniform samples in [-1,1]^3 and keep those with r^2 <= 1.
    """
    samples = np.empty((0,3))
    # propose in batches sized by expected acceptance fraction (vol(ball)/8 ~ 0.5236/8?)
    vol_cube = 8.0
    acc_frac = volume_ball(1.0) / vol_cube
    batch = max(int(N * batch_factor), 1000)
    while samples.shape[0] < N:
        X = np.random.uniform(-1.0, 1.0, size=(batch, 3))
        r2 = np.sum(X**2, axis=1)
        keep = X[r2 <= 1.0]
        if keep.size:
            samples = np.vstack((samples, keep))
    return samples[:N]

def mc_standard_estimator(f_points, N):
    """Standard Monte-Carlo estimator using N samples in the ball.


    Returns (estimate, n_evals, sample_values): sample_values are raw f evaluations (N,)
    """
    X = sample_ball_rejection(N)
    vals = f_points(X)
    estimate = volume_ball(1.0) * np.mean(vals)
    return estimate, N, vals

def symmetrized_estimator(f_points, N):
    """
    Symmetrized Monte-Carlo: for each sample x, evaluate f on the 8 sign variations
    and average them. This requires 8 f-evaluations per base sample.
    """
    X = sample_ball_rejection(N)
    per_sample_avgs = np.zeros(N)
    # iterate samples and compute average over sign flips
    for i, x in enumerate(X):
        s = 0.0
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            xi = x * np.array(signs)
            s += f_points(xi.reshape(1,3))[0]
        per_sample_avgs[i] = s / 8.0
    estimate = volume_ball(1.0) * np.mean(per_sample_avgs)
    n_evals = 8 * N
    return estimate, n_evals, per_sample_avgs

def mc_estimate_with_repeats(f_points, N_list, repeats=10, symmetrize=False):
    """
    Helper to run Monte-Carlo for several N values and repeats.
    """
    out = {}
    for N in N_list:
        estimates = []
        for r in range(repeats):
            if symmetrize:
                est, n_evals, _ = symmetrized_estimator(f_points, N)
            else:
                est, n_evals, _ = mc_standard_estimator(f_points, N)
            estimates.append(est)
        out[N] = {'estimates': np.array(estimates), 'n_evals': n_evals}
    return out

# Numerical checks: unbiasedness and sample variance
def mc_stats_from_samples(vals):
    """Returns estimator mean and variance of S_N = vol * mean(vals) 
    given an array of function values on samples inside ball."""
    vol = volume_ball(1.0)
    sample_mean = np.mean(vals)
    sample_var = np.var(vals, ddof=1)
    estimator_mean = vol * sample_mean
    estimator_var = (vol**2) * sample_var / vals.size
    return estimator_mean, estimator_var
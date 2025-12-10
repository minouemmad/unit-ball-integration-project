#src/montecarlo.py
import numpy as np
import itertools
from math import pi
from .utils import f_from_points, volume_ball

def sample_uniform_ball_direct(N, d, rng=np.random):
    """
    Generate N uniform points in the unit d-ball using:
      - sample Z ~ N(0,I) in R^d
      - u = Z / ||Z||  (uniform on sphere)
      - r = U^(1/d) where U ~ Uniform(0,1)
      - return r * u
    Returns shape (N, d)
    """
    Z = rng.normal(size=(N, d))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    # avoid division by zero (rare)
    norms[norms == 0] = 1.0
    u = Z / norms
    U = rng.random(size=(N, 1))
    r = U ** (1.0 / d)
    return u * r

def sample_ball_rejection(N, d=3, batch_factor=1.5, rng=np.random):
    """
    Rejection sampling in [-1,1]^d; return N accepted samples.
    This is only for small d because acceptance = vol(B_d)/2^d.
    """
    samples = np.empty((0,3))
    # propose in batches sized by expected acceptance fraction (vol(ball)/8 ~ 0.5236/8?)
    vol_cube = 2.0 ** d
    acc_frac = volume_ball(1.0, d=d) / vol_cube
    batch = max(int(N * batch_factor), 1000)
    samples = np.empty((0, d))
    while samples.shape[0] < N:
        X = np.random.uniform(-1.0, 1.0, size=(batch, d))
        r2 = np.sum(X**2, axis=1)
        keep = X[r2 <= 1.0]
        if keep.size:
            samples = np.vstack((samples, keep))
    return samples[:N]

def mc_standard_estimator(f_points, N, d=3, sampling='direct', rng=np.random):
    """
    Standard MC: sample N points in unit d-ball and compute vol * mean(f(points))
    sampling: direct or rejection  
    """
    if sampling == 'direct':
        X = sample_uniform_ball_direct(N, d, rng=rng)
        n_evals = N
    elif sampling == 'rejection':
        X = sample_ball_rejection(N, d=d, rng=rng)
        n_evals = N
    else:
        raise ValueError("sampling must be 'direct' or 'rejection'")

    vals = f_points(X)
    vol = volume_ball(1.0, d=d)
    estimate = vol * np.mean(vals)
    return estimate, n_evals, vals

def symmetrized_estimator(f_points, N, d=3, sampling='direct', rng=np.random, max_full_sym_d=6, subset_signs=None):
    """
    For each sample x, average f over sign flips (2^d patterns).
    Defensive: verifies sampled point dimensionality matches d; if not, it
    resets d to the sample dimension.
    """
    if sampling == 'direct':
        X = sample_uniform_ball_direct(N, d, rng=rng)
    else:
        X = sample_ball_rejection(N, d=d, rng=rng)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2-D array of shape (N,d); got X.ndim={X.ndim}")
    sample_d = X.shape[1]
    if sample_d != d:
        print(f"Warning: requested d={d} but sampler returned points with dimension {sample_d}. "
              f"Using sample dimension {sample_d} for symmetrization.")
        d = sample_d

    per_sample_avgs = np.zeros(X.shape[0], dtype=float)

    if d <= max_full_sym_d:
        signs_list = list(itertools.product([-1.0, 1.0], repeat=d))
    elif subset_signs is not None:
        signs_list = subset_signs
    else:
        k = min(32, 2 ** d)
        rng_local = rng
        signs_list = []
        for _ in range(k):
            signs_list.append(tuple(rng_local.choice([-1.0, 1.0], size=(d,)).tolist()))

    n_patterns = len(signs_list)

    for i, x in enumerate(X):
        # create stacked points for vectorized evaluation: shape (n_patterns, d)
        stacked = np.vstack([x * np.array(sign) for sign in signs_list])
        vals = f_points(stacked)  # expects (n_patterns,) array
        per_sample_avgs[i] = np.mean(vals)

    vol = volume_ball(1.0, d=d)
    estimate = vol * np.mean(per_sample_avgs)
    n_evals = n_patterns * X.shape[0]
    return estimate, n_evals, per_sample_avgs

def mc_estimate_with_repeats(f_points, N_list, repeats=10, symmetrize=False, d=3, sampling='direct', rng=np.random):
    """
    Run Monte-Carlo for several N values and repeats.
    """
    out = {}
    for N in N_list:
        estimates = []
        n_evals = None
        for r in range(repeats):
            if symmetrize:
                est, n_evals, _ = symmetrized_estimator(f_points, N, d=d, sampling=sampling, rng=rng)
            else:
                est, n_evals, _ = mc_standard_estimator(f_points, N, d=d, sampling=sampling, rng=rng)
            estimates.append(est)
        out[N] = {'estimates': np.array(estimates), 'n_evals': n_evals}
    return out

# Numerical checks: unbiasedness and sample variance
def mc_stats_from_samples(vals, d=3):
    """Returns estimator mean and variance of S_N = vol * mean(vals) 
    given an array of function values on samples inside ball."""
    vol = volume_ball(1.0, d=d)
    sample_mean = np.mean(vals)
    sample_var = np.var(vals, ddof=1)
    estimator_mean = vol * sample_mean
    estimator_var = (vol**2) * sample_var / vals.size
    return estimator_mean, estimator_var
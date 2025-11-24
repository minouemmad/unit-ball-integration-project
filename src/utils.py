"""
Helpers used by both deterministic and Monte-Carlo implementations.
For tasks 1..5.
"""
import numpy as np
from math import pi

VOL_BALL = 4.0/3.0 * pi

# -- Test integrand (assignment-specified) --
def f_xyzeval(x, y, z):
    """Vectorized evaluator for f(x,y,z) = (1 + x^2 + y^2) * exp(z) - x/(1+z^2).
    """
    return (1.0 + x**2 + y**2) * np.exp(z) - x / (1.0 + z**2)

def f_from_points(X):
    """Convenience: X is (N,3) array, returns (N,) array of f values."""
    X = np.asarray(X)
    return f_xyzeval(X[:,0], X[:,1], X[:,2])

def volume_ball(radius=1.0):
    return 4.0/3.0 * pi * radius**3

# RNG seeding helper
def set_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)

# map spherical to cartesian
def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z
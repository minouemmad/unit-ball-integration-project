"""
Helpers used by both deterministic and Monte-Carlo implementations.
For tasks 1..5.
"""
import numpy as np
from math import pi
from math import pi, gamma

# -- Generalized ball volume --
def volume_ball(radius=1.0, d=3):
    """
    Volume of the d-dimensional ball of given radius:
      vol_d = (pi)^(d/2) / Gamma(d/2 + 1) * radius^d
    Default behavior preserved: volume_ball(radius=1.0) -> d=3 (backwards compatible)
    """
    if d == 3:
        return 4.0/3.0 * pi * radius**3
    vol_unit = (pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)
    return vol_unit * (radius ** d)

# -- Test integrand (assignment-specified) --
def f_xyzeval(x, y, z):
    """Vectorized evaluator for f(x,y,z) = (1 + x^2 + y^2) * exp(z) - x/(1+z^2).
    """
    return (1.0 + x**2 + y**2) * np.exp(z) - x / (1.0 + z**2)

def f_from_points(X):
    X = np.asarray(X)
    # If X is shape (N,d) and d==3, use same integrand
    if X.ndim == 2 and X.shape[1] == 3:
        return f_xyzeval(X[:, 0], X[:, 1], X[:, 2])
    # If d != 3, user must supply compatible integrand; as a fallback, reduce to first 3 coords
    if X.ndim == 2 and X.shape[1] > 3:
        return f_xyzeval(X[:, 0], X[:, 1], X[:, 2])
    raise ValueError("f_from_points expects an (N,3) array for current integrand.")

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
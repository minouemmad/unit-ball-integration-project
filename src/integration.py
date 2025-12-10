# src/integration.py
import numpy as np
from .utils import spherical_to_cartesian


def grid_1d_uniform(a, b, m):
    """Uniform grid (m points) on [a,b]."""
    pts = np.linspace(a, b, m)
    if m > 1:
        h = (b - a) / (m - 1)
    else:
        h = (b - a)
    return pts, h

def spherical_grid_integral(f_points, m_r=60, m_theta=60, m_phi=60):
    """
    Nested 1-D grid integration over unit ball using spherical coordinates.
    Uses simple product-rule Riemann sum with uniform spacing.
    """
    # grids
    r_pts, dr = grid_1d_uniform(0.0, 1.0, m_r)
    theta_pts, dtheta = grid_1d_uniform(-np.pi/2.0, np.pi/2.0, m_theta)
    phi_pts, dphi = grid_1d_uniform(0.0, 2.0*np.pi, m_phi)

    # build mesh using broadcasting to avoid huge intermediate arrays
    R, T, P = np.meshgrid(r_pts, theta_pts, phi_pts, indexing='ij')
    x, y, z = spherical_to_cartesian(R, T, P)

    # evaluate integrand
    vals = f_points(np.vstack((x.ravel(), y.ravel(), z.ravel())).T)
    vals = vals.reshape(x.shape)

    # Jacobian
    weights = (R**2) * np.cos(T)

    integral = np.sum(vals * weights) * dr * dtheta * dphi
    n_evals = m_r * m_theta * m_phi
    return integral, n_evals

# Convenience wrapper that uses the same m for all dims
def spherical_grid_integral_equal_m(f_points, m):
    return spherical_grid_integral(f_points, m_r=m, m_theta=m, m_phi=m)
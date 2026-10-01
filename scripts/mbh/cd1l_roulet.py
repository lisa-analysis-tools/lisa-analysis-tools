"""Volume-preserving spin proposals in the 11-D CD1-L chirp-mass basis.

Only proposal coordinates change; physical phase, priors, WDM, and saved chain
basis stay unchanged. This module has no waveform/GPU dependencies.
"""
import numpy as np


def to_roulet(x):
    """Replace s1z,s2z by chi_eff,chi_diff; q=m2/m1 is column 1."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != 11 or np.any(x[..., 1] <= 0):
        raise ValueError("Expected 11-D chirp-mass rows with positive q")
    y = x.copy()
    q, a, b = x[..., 1], x[..., 2], x[..., 3]
    y[..., 2] = (a + q * b) / (1 + q)
    y[..., 3] = a - b
    return y


def from_roulet(y):
    """Inverse; original spin bounds must be checked on returned rows."""
    y = np.asarray(y, dtype=float)
    if y.shape[-1] != 11 or np.any(y[..., 1] <= 0):
        raise ValueError("Expected 11-D Roulet rows with positive q")
    x = y.copy()
    q, eff, diff = y[..., 1], y[..., 2], y[..., 3]
    x[..., 2] = eff + q * diff / (1 + q)
    x[..., 3] = eff - diff / (1 + q)
    return x


def forward_jacobian(x):
    """dy/dx including q dependence; absolute determinant is one."""
    x = np.asarray(x, dtype=float)
    if x.shape != (11,) or x[1] <= 0:
        raise ValueError("Expected one chirp-mass row")
    q, a, b = x[1:4]
    j = np.eye(11)
    j[2, 1:4] = [(b-a)/(1+q)**2, 1/(1+q), q/(1+q)]
    j[3, 2:4] = [1, -1]
    return j


def transport_covariance(cov, reference):
    j = forward_jacobian(reference)
    c = j @ np.asarray(cov) @ j.T
    return (c + c.T) / 2


def propose(x, steps, lo, hi, periodic):
    """Translate in Roulet coordinates, wrap angles, reject outside the prior.

    Invalid proposals become self-transitions. Never reflect a correlated
    Gaussian at a rectangular boundary: that is not generally symmetric.
    The off-diagonal density here is symmetric since both Jacobians have
    absolute determinant one and the Gaussian covariance is fixed.
    """
    x = np.asarray(x, dtype=float)
    base = to_roulet(x)
    y = base + steps
    valid = np.isfinite(y).all(axis=-1) & (y[..., 1] > 0)
    candidate = from_roulet(np.where(valid[..., None], y, base))
    for index, period in periodic.items():
        candidate[..., index] %= period
    valid &= np.all((candidate >= lo) & (candidate <= hi), axis=-1)
    return np.where(valid[..., None], candidate, x), valid

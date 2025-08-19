import numpy as np
from typing import Optional

from .fracdiff import fracdiff


def arfima(n: int, d: float, phi: float = 0.0, sigma: float = 1.0,
           seed: Optional[int] = None, burnin: int = 0):
    r"""
    Simulate ARFIMA(1,d,0) process: (1 - \phi L) (1-L)^d X_t = \epsilon_t

    Algorithm:
    1. Generate AR(1) process:
       (1 - \phi L) u_t = \epsilon_t  =>  u_t = \phi u_{t-1} + \epsilon_t
    2. Apply fractional filter: X_t = (1-L)^(-d) u_t
    3. Discard burn-in observations

    Parameters
    ----------
    n : int
        Sample size (final output length)
    d : float
        Fractional differencing parameter
    phi : float, default=0.0
        AR(1) coefficient
    sigma : float, default=1.0
        Innovation standard deviation
    seed : int, optional
        Random seed for reproducibility
    burnin : int, default=0
        Number of burn-in observations to discard

    Returns
    -------
    np.ndarray
        ARFIMA(1,d,0) process of length n
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = n + burnin

    # Step 1: Generate AR(1) process
    eps = np.random.normal(0, sigma, n_total)

    if abs(phi) < 1e-13:
        # Pure white noise case
        u = eps
    else:
        # Generate AR(1) with proper stationary initialization
        u = np.zeros(n_total)

        if abs(phi) < 1:
            # Stationary case
            u[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))
        else:
            # Nonstationary case
            u[0] = eps[0]

        # Generate AR(1) recursively
        for t in range(1, n_total):
            u[t] = phi * u[t-1] + eps[t]

    # Step 2: Apply fractional filter if needed
    if abs(d) < 1e-13:
        # No fractional component, return AR(1) with burn-in removed
        return u[burnin:]

    # Apply fractional filter
    x = fracdiff(u, -d)

    # Discard burn-in
    return x[burnin:]

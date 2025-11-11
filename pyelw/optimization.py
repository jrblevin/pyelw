import numpy as np
from typing import Callable, Optional, Tuple

_epsilon = np.sqrt(np.finfo(np.float64).eps)


class OptimizeResult:
    """
    Simple OptimizeResult class compatible with SciPy's interface.

    Attributes
    ----------
    x : np.float64
        The solution
    fun : np.float64
        The objective function value at the solution
    success : bool
        Whether optimization succeeded
    message : str
        Description of termination
    nfev : int
        Number of function evaluations
    nit : int
        Number of iterations
    """
    def __init__(self,
                 x: np.float64,
                 fun: np.float64,
                 success: bool,
                 message: str,
                 nfev: int,
                 nit: int):
        self.x = x
        self.fun = fun
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nit = nit


def golden_section_search(func: Callable[[np.float64], np.float64],
                          brack: Optional[Tuple[np.float64, np.float64]] = None,
                          tol: Optional[np.float64] = _epsilon,
                          maxiter: Optional[int] = 100) -> OptimizeResult:
    """
    Golden section search for minimizing 1D function, with SciPy-compatible API.

    Parameters
    ----------
    func : callable
        Objective function to minimize, taking a single np.float64 argument and returning np.float64
    brack : tuple, optional
        Bounds for optimization as (lower, upper) of np.float64. Default: (-0.5, 1.0).
    tol : np.float64, optional
        Tolerance for convergence. Default matches SciPy.
    maxiter : int, optional
        Maximum number of iterations. Default: 100.

    Returns
    -------
    OptimizeResult
        Optimization result with attributes:
        - x : np.float64 - The solution
        - fun : np.float64 - Function value at solution
        - success : bool - Whether optimization succeeded
        - message : str - Termination message
        - nfev : int - Number of function evaluations
        - nit : int - Number of iterations
    """
    # Default bounds
    if brack is None:
        xl, xr = np.float64(-0.5), np.float64(1.0)
    else:
        xl, xr = np.float64(brack[0]), np.float64(brack[1])

    # Golden ratio conjugate
    gratio = np.float64(0.61803398874989)

    # Initial function evaluations
    xlower = xl + (xr - xl) * (1 - gratio)
    xupper = xl + (xr - xl) * gratio
    vlower = func(xlower)
    vupper = func(xupper)
    iter = 0
    nfev = 2  # Already evaluated at xlower and xupper

    # Track best solution found
    best_x = xlower if vlower < vupper else xupper
    best_fun = min(vlower, vupper)

    # Main loop
    while iter < maxiter:
        iter += 1

        # Check convergence with relative tolerance
        mid_point = 0.5 * (xl + xr)
        relative_size = (xr - xl) / max(1.0, abs(mid_point))
        if relative_size <= tol:
            break

        # Golden section search step with symmetric logic
        if vlower < vupper:
            xr = xupper
            xupper = xlower
            vupper = vlower
            xlower = xl + (xr - xl) * (1 - gratio)
            vlower = func(xlower)
        else:
            xl = xlower
            xlower = xupper
            vlower = vupper
            xupper = xl + (xr - xl) * gratio
            vupper = func(xupper)

        nfev += 1

        # Update best solution tracking
        if vlower < best_fun:
            best_x = xlower
            best_fun = vlower
        if vupper < best_fun:
            best_x = xupper
            best_fun = vupper

    # Final solution selection
    x_opt = best_x
    fun_opt = best_fun

    # Convergence assessment
    success = iter < maxiter
    if success:
        message = f"Optimization terminated successfully; tolerance {tol} achieved"
    else:
        message = f"Maximum number of iterations ({maxiter}) exceeded"

    return OptimizeResult(
        x=x_opt,
        fun=fun_opt,
        success=success,
        message=message,
        nfev=nfev,
        nit=iter
    )


def robust_golden_section_search(func: Callable[[np.float64], np.float64],
                                 brack: Optional[Tuple[np.float64, np.float64]] = None,
                                 n_grid: Optional[int] = 20,
                                 tol: Optional[np.float64] = _epsilon,
                                 maxiter: Optional[int] = 100) -> OptimizeResult:
    """
    Robust golden section search with grid-based safety check.

    This function enhances golden_section_search by adding a coarse grid search
    as a safety check to avoid local minima. If the grid search finds a better
    minimum than the golden section search, re-optimization is performed in the
    region around the grid minimum.

    Algorithm:
    1. Run standard golden section search over full bounds.
    2. Run coarse grid search as safety check.
    3. If grid found better minimum, re-run golden section between
       neighboring grid points.

    Parameters
    ----------
    func : callable
        Objective function to minimize, taking a single np.float64
        argument and returning np.float64
    brack : tuple, optional
        Bounds for optimization as (lower, upper) of np.float64.
        Default: (-0.5, 1.0).
    n_grid : int, optional
        Number of grid points for safety check. Default: 20.
    tol : np.float64, optional
        Tolerance for convergence. Default matches SciPy.
    maxiter : int, optional
        Maximum number of iterations. Default: 100.

    Returns
    -------
    OptimizeResult
        Optimization results as with golden_section_search.
    """
    # Default bounds
    if brack is None:
        xl, xr = np.float64(-0.5), np.float64(1.0)
    else:
        xl, xr = np.float64(brack[0]), np.float64(brack[1])

    # Step 1: Standard golden section search
    result_gs = golden_section_search(func, brack=(xl, xr), tol=tol, maxiter=maxiter)

    # Step 2: Grid search as safety check
    grid = np.linspace(xl, xr, n_grid)
    grid_values = [func(d) for d in grid]
    best_grid_idx = np.argmin(grid_values)
    obj_grid_min = grid_values[best_grid_idx]
    nfev = result_gs.nfev + n_grid

    # Step 3: If grid found better minimum, re-run golden section search
    if obj_grid_min < result_gs.fun:
        # Re-optimize between neighboring grid points.
        # Use previous and next grid points as bounds.
        if best_grid_idx == 0:
            local_bounds = (grid[0], grid[1])
        elif best_grid_idx == n_grid - 1:
            local_bounds = (grid[n_grid - 2], grid[n_grid - 1])
        else:
            local_bounds = (grid[best_grid_idx - 1], grid[best_grid_idx + 1])

        result_retry = golden_section_search(func, brack=local_bounds,
                                             tol=tol, maxiter=maxiter)
        result_retry.nfev = nfev + result_retry.nfev
        return result_retry
    else:
        # Golden section minimum stands
        result_gs.nfev = nfev
        return result_gs

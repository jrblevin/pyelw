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

import numpy as np
from typing import Optional, Dict, Any, Tuple

from .optimization import golden_section_search, robust_golden_section_search
from .fracdiff import fracdiff


class ELW:
    """
    Exact Local Whittle estimator of Shimotsu and Phillips (2005).

    Parameters
    ----------
    bounds : tuple of float, default=(-1.0, 2.2)
        Lower and upper bounds for optimization of memory parameter d.
    mean_est : str, default='none'
        Form of mean estimation. Options:
        - 'mean': subtract sample mean (valid for d in (-1/2, 1))
        - 'init': subtract initial value (valid for d > 0)
        - 'none': no mean correction
    n_grid : int, default=20
        Number of grid points for robust optimization check. If n_grid > 0,
        performs a coarse grid search to verify the golden section search
        found the global minimum. Recommended when local minima are suspected.
        Set n_grid = 0 to use standard golden section search only (faster but
        less robust).

    Attributes
    ----------
    d_hat_ : float
        Estimated memory parameter.
    se_ : float
        Standard error of the estimate.
    ase_ : float
        Asymptotic standard error.
    n_ : int
        Sample size.
    m_ : int
        Number of frequencies used.
    objective_ : float
        Final objective function value.
    nfev_ : int
        Number of function evaluations.

    References
    ----------
    Shimotsu, K. and Phillips, P.C.B. (2005). Exact Local Whittle Estimation
    of Fractional Integration. _Annals of Statistics_ 33, 1890--1933.
    """

    def __init__(self, bounds=(-1.0, 2.2), mean_est='none', n_grid=20):
        self._default_bounds = (-1.0, 2.2)
        self._default_mean_est = 'none'

        self.bounds = bounds
        self.mean_est = mean_est
        self.n_grid = n_grid

    def objective(self, d: float, X: np.ndarray, m: int) -> float:
        """
        Exact Local Whittle objective function of Shimotsu and Phillips (2005).

        Parameters
        ----------
        d : float
            Memory parameter
        X : np.ndarray
            Time series
        m : int
            Number of frequencies to use

        Returns
        -------
        float
            ELW objective function value, to be minimized
        """
        n = len(X)

        try:
            # Fractionally difference the original series
            dx = fracdiff(X, d)

            # Compute FFT and periodogram
            fft_dx = np.fft.fft(dx)
            I_dx = np.abs(fft_dx)**2 / (2 * np.pi * n)

            # Use first m frequencies (excluding zero)
            I_dx_m = I_dx[1:m+1]  # frequencies 1, 2, ..., m
            freqs = 2 * np.pi * np.arange(1, m+1, dtype=np.float64) / n

            # ELW objective function
            G_hat = np.mean(I_dx_m)
            if G_hat <= 0:
                return np.float64(np.inf)

            first_term = np.log(G_hat)
            second_term = -2 * d * np.mean(np.log(freqs))
            obj = first_term + second_term

            if not np.isfinite(obj):
                return np.float64(np.inf)

            return np.float64(obj)

        except (OverflowError, ZeroDivisionError, ValueError):
            return np.float64(np.inf)

    def fit(self, X, m=None, verbose=False):
        """
        Exact local Whittle estimation of memory parameter d.

        Parameters
        ----------
        X : np.ndarray
            Time series data.
        m : int or 'auto', optional
            Number of frequencies to use. Options:
            - int: Use specified number of frequencies
            - None: Use default n^0.65
            - 'auto': Use bootstrap procedure to select optimal bandwidth
        verbose : bool, default=False
            Print diagnostic information during fitting.

        Returns
        -------
        self : object
            Returns the fitted estimator.
        """
        # Mean adjustment (see Shimotsu, 2010, section 3)
        if self.mean_est == 'mean':
            # Subtract sample mean
            X = X - np.mean(X)
        elif self.mean_est == 'init':
            # Subtract initial value
            X = (X - X[0])[1:]
        elif self.mean_est == 'none':
            pass
        else:
            raise ValueError("mean_est must be one of 'mean', 'init', 'none'")

        # Sample size
        n = len(X)
        if m is None:
            m = int(n**0.65)
        elif m == 'auto':
            # Use bootstrap MSE bandwidth selection to find optimal m
            from .lw_bootstrap_m import LWBootstrapM
            selector = LWBootstrapM(bounds=self.bounds, verbose=verbose)
            selector.fit(X)

            # Store bootstrap-specific attributes
            self.bootstrap_m_optimal_m_ = selector.optimal_m_
            self.bootstrap_m_iterations_ = selector.iterations_
            self.bootstrap_m_mse_profile_ = selector.mse_profile_
            self.bootstrap_m_k_n_ = selector.k_n_

            # Use the optimal m for ELW estimation
            m = selector.optimal_m_

        # ELW objective function
        def objective_func(d: float) -> float:
            return self.objective(d, X, m)

        # Optimize using golden section search with bounds
        if self.n_grid > 0:
            result = robust_golden_section_search(objective_func, brack=self.bounds, n_grid=self.n_grid)
        else:
            result = golden_section_search(objective_func, brack=self.bounds)

        if not result.success:
            if verbose:
                print(f"Warning: {result.message}")

        if not np.isfinite(result.x) or not np.isfinite(result.fun):
            d_hat = np.nan
            final_obj = np.nan
        else:
            d_hat = result.x
            final_obj = result.fun

        # Standard error based on Fisher information
        if np.isfinite(d_hat):
            try:
                # Finite difference approximation of second derivative
                dl = d_hat * 0.99
                du = d_hat * 1.01
                fl = objective_func(dl)
                fu = objective_func(du)
                d2 = 1.0e4*(fl - 2*final_obj + fu)/d_hat**2
                # Check for convexity
                if (d2 > 0):
                    se = np.sqrt(1/(m*d2))
                else:
                    se = np.nan

            except Exception:
                se = np.nan
        else:
            se = np.nan

        # Asymptotic standard error
        ase = 1 / (2 * np.sqrt(m))

        # Store fitted attributes
        self.n_ = n
        self.m_ = m
        self.d_hat_ = d_hat
        self.se_ = se
        self.ase_ = ase
        self.objective_ = final_obj
        self.nfev_ = result.nfev

        return self

    def estimate(self,
                 X: np.ndarray,
                 m = None,
                 bounds: Optional[Tuple[float, float]] = None,
                 mean_est: Optional[str] = None,
                 verbose: Optional[bool] = False) -> Dict[str, Any]:
        """
        Exact local Whittle estimation of memory parameter d.

        This method provides backward compatibility with the original API.
        For new code, consider using fit() followed by accessing fitted
        attributes directly.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        m : int or 'auto', optional
            Number of frequencies to use. Use 'auto' for bootstrap selection.
        bounds: tuple[float, float], optional
            Lower and upper bounds for golden section search.
            If provided, temporarily overrides constructor bounds.
        mean_est : str, optional
            Form of mean estimation. If provided, temporarily overrides
            constructor mean_est. One of ['mean', 'init', 'none'].
        verbose : bool, optional
            Print diagnostic information

        Returns
        -------
        Dict[str, Any]
            Dictionary with estimation results
        """
        # Temporarily store original parameters
        original_bounds = self.bounds
        original_mean_est = self.mean_est

        # Override parameters if provided
        if bounds is not None:
            self.bounds = bounds
        if mean_est is not None:
            self.mean_est = mean_est

        try:
            # Fit the model
            self.fit(X, m=m, verbose=verbose)

            # Return results as dictionary for backward compatibility
            return {
                'n': self.n_,
                'm': self.m_,
                'd_hat': self.d_hat_,
                'se': self.se_,
                'ase': self.ase_,
                'objective': self.objective_,
                'nfev': self.nfev_,
                'method': 'elw',
            }
        finally:
            # Restore original parameters
            self.bounds = original_bounds
            self.mean_est = original_mean_est

    def __repr__(self):
        """Representation showing non-default parameters."""
        params = []

        if self.bounds != self._default_bounds:
            params.append(f"bounds={self.bounds}")
        if self.mean_est != self._default_mean_est:
            params.append(f"mean_est='{self.mean_est}'")

        params_str = ", ".join(params)
        return f"ELW({params_str})"

    def __str__(self):
        return self.__repr__()

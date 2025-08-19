import numpy as np
from typing import Optional, Dict, Any, Tuple

from .optimization import golden_section_search
from .fracdiff import fracdiff


class ELW:
    """
    Exact Local Whittle estimator of Shimotsu and Phillips (2005).

    References
    ----------
    Shimotsu, K. and Phillips, P.C.B. (2005). Exact Local Whittle Estimation
    of Fractional Integration. _Annals of Statistics_ 33, 1890--1933.
    """

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

    def estimate(self,
                 X: np.ndarray,
                 m: Optional[int] = None,
                 bounds: Optional[Tuple[float, float]] = (-1.0, 2.2),
                 mean_est: Optional[str] = "none",
                 verbose: Optional[bool] = False) -> Dict[str, Any]:
        """
        Exact local Whittle estimation of memory parameter d.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        m : int, optional
            Number of frequencies to use
        bounds: tuple[float, float], optional
            Lower and upper bounds for golden section search
        mean_est : str, optional
            Form of mean estimation. One of ['mean', 'init', 'none'].
            - 'mean': subtract sample mean (valid for d in (-1/2, 1))
            - 'init': subtract initial value (valid for d > 0)
            - 'none': no mean correction
        verbose : bool, optional
            Print diagnostic information

        Returns
        -------
        Dict[str, Any]
            Dictionary with estimation results
        """

        # Mean adjustment (see Shimotsu, 2010, section 3)
        if mean_est == 'mean':
            # Subtract sample mean
            X = X - np.mean(X)
        elif mean_est == 'init':
            # Subtract initial value
            X = (X - X[0])[1:]
        elif mean_est == 'none':
            pass
        else:
            raise ValueError("mean_est must be one of 'mean', 'init', 'none'")

        # Sample size
        n = len(X)
        if m is None:
            m = int(n**0.65)

        # ELW objective function
        def objective_func(d: float) -> float:
            return self.objective(d, X, m)

        # Optimize using golden section search with bounds
        result = golden_section_search(objective_func, brack=bounds)

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

        return {
            'n': n,
            'm': m,
            'd_hat': d_hat,
            'se': se,
            'ase': ase,
            'objective': final_obj,
            'nfev': result.nfev,
            'method': 'elw',
        }

import numpy as np
from typing import Optional, Dict, Any, Tuple

from .optimization import golden_section_search
from .fracdiff import fracdiff
from .lw import LW


class TwoStepELW:
    """
    Two-step Exact Local Whittle estimator of Shimotsu (2010).

    Parameters
    ----------
    bounds : tuple of float, default=(-1.0, 2.2)
        Lower and upper bounds for optimization of memory parameter d.
    taper : str, default='hc'
        Type of taper for Stage 1 estimation. Options:
        - 'none': Standard local Whittle
        - 'kolmogorov': Zhurbenko-Kolmogorov taper (Velasco, 1999)
        - 'cosine': Cosine bell taper (Velasco, 1999)
        - 'bartlett': Triangular (Bartlett) taper (Velasco, 1999)
        - 'hc': Complex cosine bell taper (Hurvich and Chen, 2000)
    trend_order : int, default=0
        Order of polynomial detrending. 0 = demean only, 1 = remove linear trend, etc.

    Attributes
    ----------
    d_hat_ : float
        Final (Stage 2) estimated memory parameter.
    se_ : float
        Final standard error of the estimate.
    ase_ : float
        Final asymptotic standard error.
    d_step1_ : float
        Stage 1 estimated memory parameter.
    se_step1_ : float
        Stage 1 standard error.
    n_ : int
        Sample size.
    m_ : int
        Number of frequencies used.
    objective_ : float
        Final (Stage 2) objective function value.
    objective_step1_ : float
        Stage 1 objective function value.
    nfev_ : int
        Stage 2 number of function evaluations.
    nfev_step1_ : int
        Stage 1 number of function evaluations.

    References
    ----------
    Shimotsu, K. (2010). Exact Local Whittle Estimation of Fractional
    Integration with Unknown Mean and Time Trend. _Econometric Theory_ 26,
    501--540.

    Hurvich, C. M., and W. W. Chen (2000). An Efficient Taper for Potentially
    Overdifferenced Long-Memory Time Series. _Journal of Time Series Analysis_
    21, 155--180.

    Velasco, C. (1999). Gaussian Semiparametric Estimation for Non-Stationary
    Time Series. _Journal of Time Series Analysis_ 20, 87--126.
    """

    def __init__(self, bounds=(-1.0, 2.2), taper='hc', trend_order=0):
        self.bounds = bounds
        self.taper = taper
        self.trend_order = trend_order

        # Store defaults for __repr__
        self._default_bounds = (-1.0, 2.2)
        self._default_taper = 'hc'
        self._default_trend_order = 0

    def weight_function(self, d: float) -> float:
        """
        Compute weight function w(d) for adaptive mean estimation.

        Parameters
        ----------
        d : float
            Memory parameter estimate

        Returns
        -------
        float
            Weight value for adaptive mean estimation

        Notes
        -----
        Following Shimotsu (2010), the weight function is:
        w(d) = 1 for d <= 0.5 (sample mean for stationary series)
        w(d) = 0 for d >= 0.75 (first observation for persistent series)
        w(d) = (1/2)[1 + cos(-2*pi+4*pi*d)] for d in (0.5, 0.75) (smooth transition)

        This matches the original Matlab implementation in ewhittle.m:
        weight = (d<=0.5) + (1/2)*(1 + cos(-2*pi+4*pi*d))*(d>0.5)*(d<0.75)
        """
        if d <= 0.5:
            return 1.0
        elif d < 0.75:
            return 0.5 * (1.0 + np.cos(-2*np.pi + 4*np.pi*d))
        else:
            return 0.0

    def detrend(self, X: np.ndarray, order: int = 0) -> np.ndarray:
        """
        Remove time trend of specified order.

        Following Shimotsu (2010) Section 4.2, we regress X_t on
        (1, t, t^2, ..., t^k) and return residuals.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        order : int, optional
            Order of time trend to remove.  For order=0,
            we remove a constant (demean).

        Returns
        -------
        np.ndarray
            Detrended time series
        """
        if order == 0:
            return X - np.mean(X)  # Demean only

        n = len(X)

        # Create polynomial trend regressors: (1, t, t^2, ..., t^order)
        t = np.arange(1, n+1, dtype=np.float64)

        # Design matrix: each column is t^i for i = 0, 1, ..., order
        Z = np.ones((n, order + 1), dtype=np.float64)
        for i in range(1, order + 1):
            Z[:, i] = t ** i

        # OLS regression: X = Z*beta + residuals
        # beta = (Z'Z)^(-1) Z'X
        ZtZ = Z.T @ Z
        ZtX = Z.T @ X

        try:
            beta = np.linalg.solve(ZtZ, ZtX)
            X_fitted = Z @ beta
            X_detrended = X - X_fitted
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            beta = np.linalg.pinv(Z) @ X
            X_fitted = Z @ beta
            X_detrended = X - X_fitted

        return X_detrended

    def objective(self, d: float, x: np.ndarray, m: int) -> float:
        """
        Exact local Whittle objective function.

        Parameters
        ----------
        d : float
            Memory parameter
        x : np.ndarray
            Time series data (already demeaned/detrended residuals)
        m : int
            Number of frequencies to use

        Returns
        -------
        float
            Objective function value to be minimized
        """
        # Mean correction for detrended residuals following Shimotsu (2010) Section 4.2:
        # Since detrended residuals sum to zero, use simplified correction
        # \phi(d) = (1 - w(d)) X_1
        weight = self.weight_function(d)
        myu = (1 - weight) * x[0]
        x_corrected = x - myu

        # ELW objective function
        dx = fracdiff(x_corrected, d)
        n = len(dx)
        t = np.arange(0, n, dtype=np.float64)  # t = (0:1:n-1)'
        lam = 2 * np.pi * t / n  # lambda = 2*pi*t/n
        # wdx = (2*pi*n)^(-1/2)*conj(fft(conj(dx))).*exp(i*lambda)
        fft_dx = np.fft.fft(np.conj(dx))
        wdx = np.conj(fft_dx) * np.exp(1j * lam) / np.sqrt(2 * np.pi * n)
        lam_trunc = lam[1:m+1]
        vx = wdx[1:m+1]
        Iv = vx * np.conj(vx)
        g = np.sum(Iv) / m
        r = np.log(g) - 2 * d * np.sum(np.log(lam_trunc)) / m
        return float(r.real)

    def fit(self, X, m=None, verbose=False):
        """
        Two-step exact local Whittle estimation of memory parameter d.

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
            Print diagnostic information during estimation.

        Returns
        -------
        self : object
            Returns the fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        n = len(X)

        # Step 0: Detrending
        if verbose and self.trend_order > 0:
            print(f"Detrending with polynomial order {self.trend_order}")
        elif verbose:
            print("Demeaning data (detrend order 0)")
        X_detrended = self.detrend(X, self.trend_order)

        # Number of frequencies
        if m is None:
            m = int(n**0.65)
        elif m == 'auto':
            # Use bootstrap MSE bandwidth selection to find optimal m
            from .lw_bootstrap_m import LWBootstrapM
            selector = LWBootstrapM(bounds=self.bounds, verbose=verbose)
            selector.fit(X_detrended)

            # Store bootstrap-specific attributes
            self.bootstrap_m_optimal_m_ = selector.optimal_m_
            self.bootstrap_m_iterations_ = selector.iterations_
            self.bootstrap_m_mse_profile_ = selector.mse_profile_
            self.bootstrap_m_k_n_ = selector.k_n_

            # Use the optimal m for two-step ELW estimation
            m = selector.optimal_m_
        if verbose:
            print(f"Using {m} frequencies for both steps")

        # Stage 1: Tapered local Whittle estimator
        if verbose:
            print(f"Stage 1: {self.taper} tapered LW estimation")
        X_step1 = X_detrended  # Stage 1 uses detrended data
        lw = LW(bounds=self.bounds, taper=self.taper)
        lw.fit(X_step1, m=m)
        d_step1 = lw.d_hat_
        se_step1 = lw.se_
        objective_step1 = lw.objective_
        nfev_step1 = lw.nfev_
        if verbose:
            print(f"  Stage 1 estimate: d = {d_step1:.4f}")

        # Stage 2: Modified ELW estimation
        if verbose:
            print("Stage 2: Exact local whittle estimation")
            print(f"    Starting from Stage 1: d = {d_step1:.6f}")

        def step2_objective_func(d: float) -> float:
            return self.objective(d, X_detrended, m)

        # Use narrower bounds around the initial estimate
        local_bounds = (max(self.bounds[0], d_step1 - 2.576*se_step1), min(self.bounds[1], d_step1 + 2.576*se_step1))
        result_step2 = golden_section_search(step2_objective_func, brack=local_bounds)
        d_step2 = result_step2.x
        if verbose:
            print(f"    Final estimate: d = {d_step2:.4f}")

        # Two-Step ELW standard error
        se = 1 / (2 * np.sqrt(m))

        # Store fitted attributes
        self.n_ = n
        self.m_ = m
        self.d_hat_ = d_step2
        self.se_ = se
        self.ase_ = se
        self.d_step1_ = d_step1
        self.se_step1_ = se_step1
        self.objective_ = result_step2.fun
        self.objective_step1_ = objective_step1
        self.nfev_ = result_step2.nfev
        self.nfev_step1_ = nfev_step1

        return self

    def estimate(self, X: np.ndarray,
                 m = None,
                 bounds: Optional[Tuple[float, float]] = None,
                 taper: Optional[str] = None,
                 trend_order: Optional[int] = None,
                 verbose: Optional[bool] = False) -> Dict[str, Any]:
        """
        Two-step exact local Whittle estimation of memory parameter d.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        m : int or 'auto', optional
            Number of frequencies to use. Use 'auto' for bootstrap selection. Default m = n^0.65.
        bounds: tuple[float, float], optional
            Lower and upper bounds for golden section search.
            If provided, temporarily overrides constructor bounds.
        taper : str, optional
            Type of taper for Stage 1. If provided, temporarily overrides
            constructor taper.
        trend_order : int, optional
            Order of polynomial detrending. If provided, temporarily
            overrides constructor trend_order.
        verbose : bool, optional
            Print diagnostic information

        Returns
        -------
        Dict[str, Any]
            Two-step ELW estimation results
        """
        # Temporarily store original parameters
        original_bounds = self.bounds
        original_taper = self.taper
        original_trend_order = self.trend_order

        # Override parameters if provided
        if bounds is not None:
            self.bounds = bounds
        if taper is not None:
            self.taper = taper
        if trend_order is not None:
            self.trend_order = trend_order

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
                'method': '2elw',
                'taper': self.taper,
                'd_step1': self.d_step1_,
                'se_step1': self.se_step1_,
                'nfev_step1': self.nfev_step1_,
                'objective_step1': self.objective_step1_,
                'nfev': self.nfev_,
                'objective': self.objective_,
                'trend_order': self.trend_order,
            }
        finally:
            # Restore original parameters
            self.bounds = original_bounds
            self.taper = original_taper
            self.trend_order = original_trend_order

    def __repr__(self):
        """Representation showing non-default parameters."""
        params = []

        if self.bounds != self._default_bounds:
            params.append(f"bounds={self.bounds}")
        if self.taper != self._default_taper:
            params.append(f"taper='{self.taper}'")
        if self.trend_order != self._default_trend_order:
            params.append(f"trend_order={self.trend_order}")

        params_str = ", ".join(params)
        return f"TwoStepELW({params_str})"

    def __str__(self):
        return self.__repr__()

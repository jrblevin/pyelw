import numpy as np
from typing import Optional, Dict, Any, Tuple

from .optimization import golden_section_search


class LW:
    """
    Standard and tapered local Whittle estimation.

    Implements the standard local Whittle estimator of Robinson (1995)
    as well as the tapered local Whittle estimators of Velasco (1999) and
    Hurvich and Chen (2000).

    Parameters
    ----------
    bounds : tuple of float, default=(-1.0, 2.2)
        Lower and upper bounds for optimization of memory parameter d.
    taper : str, default='none'
        Type of taper to apply. Options:
        - 'none': No taper (standard local Whittle)
        - 'kolmogorov': Zhurbenko-Kolmogorov taper (Velasco, 1999)
        - 'cosine': Cosine bell taper (Velasco, 1999)
        - 'bartlett': Triangular (Bartlett) taper (Velasco, 1999)
        - 'hc': Complex cosine bell taper of Hurvich and Chen (2000)

    diff : int, default=1
        Number of times to difference for HC taper. Only used when taper='hc'.

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
    method_ : str
        Estimation method used
    taper_ : str
        Actual taper used during estimation.
    diff_ : int
        Number of differences applied (for taper='hc').

    References
    ----------
    Robinson, P. M. (1995). Gaussian Semiparametric Estimation of Long
    Range Dependence. _Annals of Statistics_ 23, 1630--1661.

    Velasco, C. (1999). Gaussian Semiparametric Estimation for Non-Stationary
    Time Series. _Journal of Time Series Analysis_ 20, 87--126.

    Hurvich, C. M., and W. W. Chen (2000). An Efficient Taper for Potentially
    Overdifferenced Long-Memory Time Series. _Journal of Time Series Analysis_
    21, 155--180.
    """

    def __init__(self, bounds=(-1.0, 2.2), taper='none', diff=1):
        self._default_bounds = (-1.0, 2.2)
        self._default_taper = 'none'
        self._default_diff = 1

        self.bounds = bounds
        self.taper = taper
        self.diff = diff

    def _hc_dft(self, y: np.ndarray, max_j: Optional[int] = None):
        r"""
        Compute DFT using Hurvich and Chen (2000) definition:
        w_j = 1/sqrt(2 \pi n) \sum_{t=1}^n y_t \exp(i \lambda_j t)
        where \lambda_j = 2 \pi j/n and y_t = h_t x_t is the tapered
        series.

        Parameters
        ----------
        y : np.ndarray
            Input data
        max_j : int, optional
            Maximum frequency index to compute. If None, computes up to n.

        Returns
        -------
        np.ndarray
            Complex DFT values w_j for j = 0, 1, ..., max_j
        """
        n = len(y)

        if max_j is None:
            max_j = n - 1

        # Create frequency indices and time indices
        j = np.arange(max_j + 1)
        t = np.arange(1, n + 1)

        # Compute lambda_j matrix: 2 * pi * j / n for each j
        lambda_j = 2 * np.pi * j[:, np.newaxis] / n

        # Compute complex exponentials for all (j, t) combinations
        # Multiply by data and sum over t
        w = np.sum(y * np.exp(1j * lambda_j * t), axis=1)

        # Normalize
        w /= np.sqrt(2 * np.pi * n)

        return w

    def prepare_data(self, X: np.ndarray, m: int, taper: Optional[str] = 'none',
                     diff: Optional[int] = 1) -> Dict[str, np.ndarray]:
        """
        Precompute quantities used for local Whittle estimation.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        m : int
            Number of frequencies to use in estimation
        taper : str, optional
            Type of taper to apply.
        diff : int, optional
            Number of times to difference data (only for 'hc' taper)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing precomputed data for estimation:
            - For 'none': 'n', 'm', 'I_X' (periodogram), 'freqs'
            - For 'kolmogorov', 'cosine', and 'bartlett': 'n', 'm', 'I_X' (subsampled periodogram), 'freqs', 'p', 'Phi'
            - For 'hc': 'n', 'm', 'I_X' (periodogram), 'freqs', 'diff'
        """
        n = len(X)

        # Apply taper and differencing if needed to produce X_tapered
        if taper == 'kolmogorov':

            # Zhurbenko-Kolmogorov taper (Velasco, 1999, p. 97)
            # Implementation follows convolution calculation by Shimotsu in veltaper.m
            pp = int((n + 2) / 3)
            h = np.ones(pp)
            h2 = np.concatenate([np.arange(1, pp + 1), np.arange(pp - 1, 0, -1)])
            h3 = np.convolve(h, h2)
            # Pad with zeros if needed
            if len(h3) < n:
                h = np.concatenate([h3, np.zeros(n - len(h3))])
            else:
                h = h3[:n]
            X_tapered = h * X

        elif taper == 'cosine':

            # Cosine bell taper (Velasco, 1999, p. 101)
            t = np.arange(1, n+1, dtype=np.float64) / n
            h = 0.5 * (1 - np.cos(2 * np.pi * t))
            X_tapered = h * X

        elif taper == 'bartlett':

            # Triangular (Bartlett) taper
            mm = np.ceil(n/2)
            t = np.arange(n, dtype=np.float64)
            h = 1 - np.abs(t + 1 - mm) / mm
            X_tapered = h * X

        elif taper == 'hc':

            # Hurvich and Chen (2000) difference the data first
            X_diff = X.copy()
            for _ in range(diff):
                X_diff = np.diff(X_diff)
            n = len(X_diff)  # Update n after differencing

            # Apply complex cosine bell taper from Hurvich-Chen (2000, eq. 3):
            # h_t = 0.5*(1 - exp(i * 2 * pi * (t - 1/2) / n)).
            # This creates a complex-valued taper, multiplied with real data
            # to produce complex-valued tapered data (eq. 4).
            #
            t = np.arange(1, n+1, dtype=np.float64)
            h = 0.5 * (1 - np.exp(1j * 2 * np.pi * (t - 0.5) / n))
            X_tapered = h * X_diff

        else:

            X_tapered = X

        # For most tapers (and untapered), use NumPy's FFT, but for HC, we
        # use the paper's DFT directly.
        if taper != 'hc':
            fft_X = np.fft.fft(X_tapered)

        if taper in ['kolmogorov', 'cosine', 'bartlett']:

            # For Velasco (1999) tapers, first compute full periodogram (skip DC)
            I_X_full = (np.abs(fft_X[1:])**2) / (2 * np.pi * n)

            # Subsample with appropriate step p
            if taper == 'bartlett':
                p = 2  # Bartlett is equivalent to Zhurbenko p=2
            else:
                p = 3  # Kolmogorov and cosine use p=3

            j = np.arange(p, m+1, p)  # 1-based indices
            freqs = 2 * np.pi * j / n

            # Subsample periodogram and compute frequencies
            indices = j - 1  # Convert to 0-based for array access
            I_X = I_X_full[indices]  # Pre-subsampled periodogram

            return {
                'n': n,
                'm': m,
                'I_X': I_X,  # Subsampled periodogram
                'freqs': freqs,
                'p': p,
                'Phi': np.float64(1.00354) if taper == 'kolmogorov' else np.float64(1.05000) if taper == 'bartlett' else np.float64(1.0),  # p. 101
            }

        elif taper == 'hc':

            # Compute HC's DFT of the tapered data
            w_tapered = self._hc_dft(X_tapered, max_j=m)

            # Normalization: for HC taper, \sum_t |h_t|^2 = n/2.
            # The paper's DFT already includes 1/sqrt(2 \pi n),
            # so we need to adjust by sqrt(n) / sqrt(n/2) = sqrt(2).
            w_T = w_tapered[1:m+1] * np.sqrt(2)

            # Tapered periodogram I_j^T = |w_j^T|^2
            I_X = np.abs(w_T)**2

            # HC frequencies
            j_tilde = np.arange(1, m+1) + 0.5
            freqs = 2 * np.pi * j_tilde / n

            return {
                'n': n,
                'm': m,
                'I_X': I_X,
                'freqs': freqs,
                'diff': diff,
            }

        else:

            # Standard periodogram (skip DC component)
            I_X = np.abs(fft_X[1:m+1])**2 / (2 * np.pi * n)

            # Frequencies: \lambda_j = (2 \pi j)/n for j = 1, ..., m
            freqs = 2 * np.pi * np.arange(1, m+1, dtype=np.float64) / n

            return {
                'n': n,
                'm': m,
                'I_X': I_X,
                'freqs': freqs,
            }

    def objective(self, d: float, data: Dict[str, np.ndarray]) -> float:
        r"""
        Robinson (1995) local Whittle objective function.

        Implements the local Whittle objective function from Robinson (1995),
        parameterized in terms of the fractional integration parameter d rather
        than the Hurst parameter H.

        The original Robinson (1995) objective function is:

        R(H) = \log \hat{G}(H) - (2H-1) \frac{1}{m} \sum_{j=1}^m \log \lambda_j
        where \hat{G}(H) = \frac{1}{m} \sum_{j=1}^m \lambda_j^{2H-1} I_j

        This implementation uses the fractional integration parameter d, where
        H = d + 0.5.  Substituting this relationship yields the equivalent
        formulation:

        K(d) = \log\left(\frac{1}{m} \sum_{j=1}^m I_j \lambda_j^{2d}\right)
             - \frac{2d}{m} \sum_{j=1}^m \log(\lambda_j)

        Parameters
        ----------
        d : float
            Fractional integration parameter
        data : Dict[str, np.ndarray]
            Precomputed quantities from prepare_data

        Returns
        -------
        float
            Local Whittle objective value to be minimized
        """
        freqs = data['freqs']
        I_X = data['I_X']

        try:
            G_hat = np.mean(I_X * (freqs**(2*d)))
            if G_hat <= 0:
                return np.float64(np.inf)

            obj = np.log(G_hat) - 2 * d * np.mean(np.log(freqs))
            if not np.isfinite(obj):
                return np.float64(np.inf)

            return np.float64(obj)

        except (OverflowError, ZeroDivisionError, ValueError):
            return np.float64(np.inf)

    def objective_velasco(self, d: float, data: Dict[str, np.ndarray]) -> float:
        """
        Velasco (1999) tapered local Whittle objective function.

        See R_p(d) on p. 99 of Velasco (1999) for details.

        Parameters
        ----------
        d : float
            Memory parameter
        data : Dict[str, np.ndarray]
            Precomputed quantities from prepare_data

        Returns
        -------
        float
            Objective function value to be minimized
        """
        try:
            I_X = data['I_X']  # Already subsampled in prepare_data
            freqs = data['freqs']
            p = data['p']
            m = data['m']

            if len(I_X) == 0:
                return np.float64(np.inf)

            # Velasco objective function with pre-subsampled data
            G_hat = (p / m) * np.sum(I_X * (freqs**(2*d)))
            if G_hat <= 0:
                return np.float64(np.inf)
            obj = np.log(G_hat) - 2*d*(p/m) * np.sum(np.log(freqs))

            if not np.isfinite(obj):
                return np.float64(np.inf)
            return np.float64(obj)

        except (OverflowError, ZeroDivisionError, ValueError, KeyError):
            return np.float64(np.inf)

    def objective_hc(self, d: float, data: Dict[str, np.ndarray]) -> float:
        """
        Hurvich and Chen (2000) tapered local Whittle objective function.

        Implements R(d^*) on page 160.

        Parameters
        ----------
        d : float
            Memory parameter (for differenced data)
        data : Dict[str, np.ndarray]
            Precomputed quantities from prepare_data

        Returns
        -------
        float
            Objective function value to be minimized
        """
        try:
            # Retrieve precomputed quantities
            I_X = data['I_X']
            freqs = data['freqs']

            # In Section 6 (simulation results), Hurvich and Chen use a modified
            # objective function with \bar{g}(\lambda) = G\{2\sin(\lambda/2)\}^{-2d^*},
            # instead of G\lambda^{-2d^*}, for ARFIMA compatibility:
            #
            # freqs = 2 * np.sin(freqs / 2)

            G_hat = np.mean(I_X * (freqs**(2*d)))
            if G_hat <= 0:
                return np.float64(np.inf)
            obj = np.log(G_hat) - 2*d * np.mean(np.log(freqs))

            if not np.isfinite(obj):
                return np.float64(np.inf)
            return np.float64(obj)

        except (OverflowError, ZeroDivisionError, ValueError, KeyError):
            return np.float64(np.inf)

    def fit(self, X, m=None, verbose=False):
        """
        Local Whittle estimation of memory parameter d.

        Parameters
        ----------
        X : np.ndarray
            Time series data
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
        X = np.asarray(X, dtype=np.float64).flatten()

        # Setup data
        n = len(X)
        if m is None:
            m = int(n**0.65)
        elif m == 'auto':
            # Use bootstrap MSE bandwidth selection to find optimal m
            from .lw_bootstrap_m import LWBootstrapM

            # Create a plain LW estimator for bootstrap (no taper)
            # Bootstrap is only defined for standard LW
            selector = LWBootstrapM(bounds=self.bounds, verbose=verbose)
            selector.fit(X)

            # Store bootstrap-specific attributes
            self.bootstrap_m_optimal_m_ = selector.optimal_m_
            self.bootstrap_m_iterations_ = selector.iterations_
            self.bootstrap_m_mse_profile_ = selector.mse_profile_
            self.bootstrap_m_k_n_ = selector.k_n_

            # Use the optimal m with current estimator's settings
            m = selector.optimal_m_

        # Prepare data with taper and differencing
        data = self.prepare_data(X, m, self.taper, self.diff)

        # Define objective function based on taper type
        if self.taper == 'none':
            method = 'lw'

            def objective_func(d: float) -> float:
                return self.objective(d, data)

        elif self.taper in ['kolmogorov', 'cosine', 'bartlett']:
            method = 'lw_velasco'

            def objective_func(d: float) -> float:
                return self.objective_velasco(d, data)

        elif self.taper == 'hc':
            method = 'lw_hc'

            # Adjust bounds for differencing
            bounds = (self.bounds[0] - self.diff, self.bounds[1] - self.diff)

            def objective_func(d: float) -> float:
                return self.objective_hc(d, data)

        else:
            raise ValueError(f"Unknown taper type: {self.taper}. "
                             "Supported: 'none', 'kolmogorov', 'cosine', 'bartlett', 'hc'")

        # Use golden section search with bounds
        if self.taper == 'hc':
            result = golden_section_search(objective_func, brack=bounds)
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
            # For HC, add diff to 'undo' first differencing
            if self.taper == 'hc':
                d_hat = d_hat + self.diff
            final_obj = result.fun

        # Standard errors
        if np.isfinite(d_hat):

            if self.taper == 'none':

                # Standard local Whittle standard error
                freqs = data['freqs']
                I_X = data['I_X']

                # Derivatives at estimated d_hat
                lambda_2d = freqs**(2 * d_hat)
                log_lambda = np.log(freqs)
                d0 = np.mean(lambda_2d * I_X)
                d1 = 2 * np.mean(log_lambda * lambda_2d * I_X)
                d2 = 4 * np.mean((log_lambda**2) * lambda_2d * I_X)

                # Fisher Information-based standard error:
                # d0/(sqrt(m)*sqrt(d0*d2-d1^2))
                fisher_info = d0 * d2 - d1**2
                if fisher_info > 0 and d0 > 0:
                    se = d0 / (np.sqrt(data['m']) * np.sqrt(fisher_info))
                else:
                    se = np.nan

                # Asymptotic standard error
                ase = 1 / (2 * np.sqrt(m))

            elif self.taper in ['kolmogorov', 'cosine', 'bartlett']:

                # Velasco (1999, p. 100)
                p = data['p']
                Phi = data['Phi']
                ase = np.sqrt(p * Phi / (4 * m))
                se = ase

            elif self.taper == 'hc':

                # Asymptotic standard errors: Hurvich and Chen (2020, Theorem 2)
                ase = np.sqrt(1.5 / (4 * m))
                # Standard errors: Hurvich and Chen (2020, eq. 11)
                two_sin_half_freqs = 2.0 * np.sin(data['freqs'] / 2.0)
                v = np.log(two_sin_half_freqs) - np.mean(np.log(two_sin_half_freqs))
                v2_sum = np.sum(v**2)
                se = np.sqrt(1.5 / (4 * v2_sum))

        else:

            se = np.nan
            ase = np.nan

        # Store fitted attributes
        self.n_ = data['n']
        self.m_ = data['m']
        self.d_hat_ = d_hat
        self.se_ = se
        self.ase_ = ase
        self.objective_ = final_obj
        self.nfev_ = result.nfev
        self.method_ = method
        self.taper_ = self.taper
        self.diff_ = self.diff if self.taper == 'hc' else 0

        return self

    def estimate(self,
                 X: np.ndarray,
                 m = None,
                 bounds: Optional[Tuple[float, float]] = None,
                 taper: Optional[str] = None,
                 diff: Optional[int] = 1,
                 verbose: Optional[bool] = False) -> Dict[str, Any]:
        """
        Local Whittle estimation of memory parameter d.

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
        taper : str, optional
            Type of taper. If provided, temporarily overrides constructor taper.
            Options: 'none', 'kolmogorov', 'cosine', 'bartlett', 'hc'
        diff : int, optional
            Number of times to difference for HC taper (default 1)
        verbose : bool, optional
            Print diagnostic information

        Returns
        -------
        Dict[str, Any]
            Dictionary with estimation results
        """
        # Temporarily store original parameters
        original_bounds = self.bounds
        original_taper = self.taper
        original_diff = self.diff

        # Override parameters if provided
        if bounds is not None:
            self.bounds = bounds
        if taper is not None:
            self.taper = taper
        if diff is not None:
            self.diff = diff

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
                'method': self.method_,
                'taper': self.taper_,
                'diff': self.diff_,
            }
        finally:
            # Restore original parameters
            self.bounds = original_bounds
            self.taper = original_taper
            self.diff = original_diff

    def __repr__(self):
        """Representation showing non-default parameters."""
        params = []

        if self.bounds != self._default_bounds:
            params.append(f"bounds={self.bounds}")
        if self.taper != self._default_taper:
            params.append(f"taper='{self.taper}'")
        if self.diff != self._default_diff:
            params.append(f"diff={self.diff}")

        params_str = ", ".join(params)
        return f"LW({params_str})"

    def __str__(self):
        return self.__repr__()

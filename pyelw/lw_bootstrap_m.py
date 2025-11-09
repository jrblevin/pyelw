import numpy as np
from typing import Optional, Tuple
from .optimization import golden_section_search


class LWBootstrapM:
    """
    Local Whittle estimator with bootstrap MSE bandwidth selection.

    Implements the local bootstrap procedure of Arteche and Orbe (2016, 2017)
    for MSE-optimal bandwidth selection.

    Parameters
    ----------
    lw_estimator : LW, optional
        Instance of LW estimator to use for estimation. If None, creates a
        default LW estimator with specified bounds.
    k_n : int or str, default='auto'
        Resampling width for local bootstrap. If 'auto', selects based on data
        characteristics. Otherwise, a positive integer <= n/2.
    B : int, default=200
        Number of bootstrap replications
    m_min : int, optional
        Minimum bandwidth to consider. Default is 6.
    m_max : int, optional
        Maximum bandwidth to consider. Default is n/2.
    m_init : int, optional
        Initial bandwidth for iteration. Default is 15.
    delta : float, default=-0.01
        Convergence criterion for iterative procedure
    max_iter : int, default=10
        Maximum number of iterations
    bounds : tuple of float, default=(-1.0, 2.2)
        Lower and upper bounds for optimization of memory parameter d. Only
        used if lw_estimator is None (when creating own LW estimator).
    verbose : bool, default=False
        Print progress information

    Attributes
    ----------
    optimal_m_ : int
        Selected optimal bandwidth
    d_hat_ : float
        Memory parameter estimate using optimal bandwidth
    se_ : float
        Standard error of the estimate
    ase_ : float
        Asymptotic standard error
    mse_profile_ : dict
        Bootstrap MSE values for each evaluated bandwidth
    k_n_ : int
        Actual resampling width used (useful when k_n='auto')
    iterations_ : int
        Number of iterations performed
    converged_ : bool
        Whether the iterative procedure converged
    n_ : int
        Sample size
    objective_ : float
        Final objective function value
    method_ : str
        Method identifier ('bootstrap_local')

    References
    ----------
    Arteche, J. and J. Orbe (2016). A bootstrap approximation for the
    distribution of the local Whittle estimator. _Computational Statistics
    and Data Analysis_ 100, 645-660.

    Arteche, J. and J. Orbe (2017). A strategy for optimal bandwidth selection
    in local Whittle estimation. _Econometrics and Statistics_ 4, 3-17.
    """

    def __init__(self,
                 lw_estimator=None,
                 k_n='auto',
                 B=200,
                 m_min=None,
                 m_max=None,
                 m_init=None,
                 delta=-0.01,
                 max_iter=10,
                 bounds=(-1.0, 2.2),
                 verbose=False):
        self.lw_estimator = lw_estimator
        self.k_n = k_n
        self.B = B
        self.m_min = m_min
        self.m_max = m_max
        self.m_init = m_init
        self.delta = delta
        self.max_iter = max_iter
        self.bounds = bounds
        self.verbose = verbose

        # Default LW estimator if none provided
        if self.lw_estimator is None:
            from . import LW  # Avoid circular import
            self.lw_estimator = LW(bounds=self.bounds)

    def _locally_standardized_periodogram(self,
                                          X: np.ndarray,
                                          d_hat: float) -> np.ndarray:
        r"""
        Compute locally standardized periodogram v_j^(1) = I_j * \lambda_j^(2d).

        Implements part of Step 1 from Arteche and Orbe (2017).

        Parameters
        ----------
        X : np.ndarray
            Time series data
        d_hat : float
            Estimated memory parameter

        Returns
        -------
        np.ndarray
            Locally standardized periodogram values for j=1,...,floor(n/2)
        """
        n = len(X)

        # Compute periodogram (skip DC component)
        fft_X = np.fft.fft(X)
        I_X = np.abs(fft_X[1:])**2 / (2 * np.pi * n)

        # Compute frequencies \lambda_j = 2*pi*j/n for j=1,2,...
        freqs = 2 * np.pi * np.arange(1, len(I_X) + 1) / n

        # Locally standardize: v_j = I_j * \lambda_j^(2d)
        v_j = I_X * (freqs**(2 * d_hat))

        return v_j[:n//2]  # Return up to Nyquist frequency

    def _spectral_flatness(self, v_j: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy) of periodogram.

        Spectral flatness measures how noise-like vs. tonal a signal is.
        It is the ratio of geometric mean to arithmetic mean, bounded [0,1].

        Parameters
        ----------
        v_j : np.ndarray
            Periodogram values (typically locally standardized)

        Returns
        -------
        float
            Spectral flatness in [0, 1]. Values near 1 indicate flat/noise-like
            spectrum, values near 0 indicate structured/tonal spectrum.

        Notes
        -----
        Uses first 100 frequencies for assessment. Computation in log domain
        for numerical stability to avoid underflow in geometric mean.
        """
        # Use first portion of spectrum for assessment
        num_freqs = min(100, len(v_j))
        v_subset = v_j[:num_freqs]

        # Compute spectral flatness using log domain for numerical stability
        # SF = exp(mean(log(x))) / mean(x) = geometric_mean / arithmetic_mean
        eps = 1e-10  # Avoid log(0)
        log_geometric_mean = np.mean(np.log(v_subset + eps))
        geometric_mean = np.exp(log_geometric_mean)
        arithmetic_mean = np.mean(v_subset)
        spectral_flatness = geometric_mean / arithmetic_mean
        return spectral_flatness

    def _select_k_n(self, v_j: np.ndarray, n: int) -> int:
        """
        Automatically select resampling width k_n based on spectral flatness.

        Following Arteche and Orbe (2017) Remark 3: "the more marked its form
        (more different from the periodogram of a white noise) the lower k_n
        should be chosen."

        Uses spectral flatness (Wiener entropy) to quantify deviation from
        white noise. Spectral flatness = geometric_mean / arithmetic_mean
        is near 1 for flat (white noise-like) spectra and near 0 for
        structured spectra with peaks.

        Parameters
        ----------
        v_j : np.ndarray
            Locally standardized periodogram
        n : int
            Sample size

        Returns
        -------
        int
            Selected resampling width

        Notes
        -----
        Spectral flatness is a standard signal processing metric for
        distinguishing noise-like signals (high flatness) from tonal/
        structured signals (low flatness). This implementation uses
        thresholds calibrated to the paper's Monte Carlo examples.
        """
        # Compute spectral flatness
        spectral_flatness = self._spectral_flatness(v_j)

        # Map spectral flatness to k_n following paper's guideline:
        # More structure (low flatness) -> smaller k_n
        # Less structure (high flatness, white noise-like) -> larger k_n
        if spectral_flatness < 0.5:  # High structure
            k_n = min(4, n // 10)
        elif spectral_flatness < 0.8:  # Moderate structure
            k_n = min(n // 10, 50)
        else:  # Low structure (close to white noise)
            k_n = min(2 * n // 5, 255)

        return k_n

    def _local_bootstrap_resample(self,
                                  v_j: np.ndarray,
                                  k_n: int,
                                  m: int,
                                  seed: Optional[int] = None) -> np.ndarray:
        """
        Perform local bootstrap resampling of standardized periodogram.

        Implements Steps 3-4 from Arteche and Orbe (2017):
        - Step 3: Generate S_j from {-k_n, ..., 0, ..., k_n} for j = 1, ..., m
        - Step 4: Resample as v*_j = v_{|j+S_j|} if |j+S_j| > 0, else v_1

        Parameters
        ----------
        v_j : np.ndarray
            Locally standardized periodogram (0-indexed array)
        k_n : int
            Resampling width
        m : int
            Number of frequencies to resample
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        np.ndarray
            Bootstrap sample of standardized periodogram
        """
        # Use local random state for reproducibility
        rng = np.random.RandomState(seed) if seed is not None else np.random

        # Step 3: Generate random shifts S_j from integers -k_n to k_n
        S = rng.randint(-k_n, k_n + 1, size=m)

        # Step 4: Apply local resampling to generate bootstrap series
        v_star = np.zeros(m)
        for j in range(m):
            # Indexing conversion: Paper uses 1-based, code uses 0-based
            # Paper's frequency j=1,2,...,m corresponds to code's j=0,1,...,m-1

            # Calculate j + S_j in paper's 1-based notation
            j_paper = j + 1
            idx_paper = j_paper + S[j]

            # Step 4: Apply formula v*_j = v_{|j+S_j|} if |j+S_j| > 0, else v_1
            # The absolute value |j+S_j| creates reflection symmetry at zero
            if idx_paper == 0:
                # Special case: j + S_j = 0, use v_1 (first frequency)
                idx_code = 0
            else:
                # Use absolute value to get frequency index, then convert to 0-based
                # Example: if j+S_j = -2, use v_{|-2|} = v_2, which is v_j[1] in code
                idx_code = abs(idx_paper) - 1

            # Handle boundary case where index exceeds available data
            if idx_code >= len(v_j):
                idx_code = len(v_j) - 1

            v_star[j] = v_j[idx_code]

        return v_star

    def _compute_bootstrap_mse(self,
                               X: np.ndarray,
                               m_eval: int,
                               d_init: float,
                               k_n: int) -> Tuple[float, np.ndarray]:
        """
        Compute bootstrap MSE for a given bandwidth.

        Implements Steps 3-7 from Arteche and Orbe (2017) for a single bandwidth.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        m_eval : int
            Bandwidth to evaluate
        d_init : float
            Current d estimate (d_1 from Step 1)
        k_n : int
            Resampling width (from Step 2)

        Returns
        -------
        float
            Bootstrap MSE estimate
        np.ndarray
            Bootstrap d estimates
        """
        n = len(X)

        # Step 1: Compute locally standardized periodogram with current d
        v_j = self._locally_standardized_periodogram(X, d_init)

        # Generate B bootstrap samples and estimates
        d_star = np.zeros(self.B)

        for b in range(self.B):
            # Steps 3-4: Resample standardized periodogram
            v_star = self._local_bootstrap_resample(v_j, k_n, m_eval, seed=b)

            # Step 5: Generate bootstrap periodogram I*_j = lambda_j^(-2d_1) v*_j
            freqs = 2 * np.pi * np.arange(1, m_eval + 1) / n
            I_star = v_star * (freqs**(-2 * d_init))

            # Prepare data for LW estimation
            data = {
                'n': n,
                'm': m_eval,
                'I_X': I_star,
                'freqs': freqs,
            }

            # Step 6: Obtain bootstrap LW estimate d*_b by minimizing R(d)
            def objective_func(d: float) -> float:
                return self.lw_estimator.objective(d, data)

            result = golden_section_search(objective_func,
                                           brack=self.lw_estimator.bounds)
            d_star[b] = result.x if result.success else np.nan

        # Step 7: Compute MSE*(m) = (1/B) \sum (d*_b(m) - d_1)^2
        valid_estimates = d_star[~np.isnan(d_star)]
        if len(valid_estimates) > 0:
            mse = np.mean((valid_estimates - d_init)**2)
        else:
            mse = np.inf

        return mse, d_star

    def fit(self, X: np.ndarray, verbose: Optional[bool] = None):
        """
        Select optimal bandwidth using the iterative bootstrap MSE
        minimization algorithm of Arteche and Orbe (2016, 2017).

        Parameters
        ----------
        X : np.ndarray
            Time series data
        verbose : bool, optional
            Override verbosity setting for this fit

        Returns
        -------
        self : object
            Returns the fitted selector with attributes set
        """
        if verbose is None:
            verbose = self.verbose

        X = np.asarray(X, dtype=np.float64).flatten()
        n = len(X)

        # Initialize parameters
        m_min = self.m_min if self.m_min is not None else 6
        m_max = min(self.m_max if self.m_max is not None else n // 2, n // 2)

        # Ensure m_min doesn't exceed m_max (can happen with small samples)
        m_min = min(m_min, m_max)

        m_init = self.m_init if self.m_init is not None else min(15, m_max)
        # Ensure m_init is within [m_min, m_max]
        m_init = max(m_min, min(m_init, m_max))

        # Initialize for iteration
        m_current = m_init
        mse_history = []
        converged = False

        # Step 2: Select resampling width k_n
        if self.k_n == 'auto':
            # Get initial d estimate
            self.lw_estimator.fit(X, m=m_current)
            d_init = self.lw_estimator.d_hat_
            v_j = self._locally_standardized_periodogram(X, d_init)
            k_n = self._select_k_n(v_j, n)
            if verbose:
                spectral_flatness = self._spectral_flatness(v_j)
                print(f"Spectral flatness = {spectral_flatness:.4f}")
                print(f"Auto-selected k_n = {k_n}")
        else:
            k_n = min(self.k_n, n // 2)

        # Store actual k_n used
        self.k_n_ = k_n

        # Iterative procedure
        for iteration in range(self.max_iter):
            if verbose:
                print(f"\nIteration {iteration + 1}")
                print(f"Current bandwidth: {m_current}")

            # Step 1: Obtain d_1 with current bandwidth m_1
            self.lw_estimator.fit(X, m=m_current)
            d_current = self.lw_estimator.d_hat_

            if verbose:
                print(f"Current d estimate: {d_current:.4f}")

            # Steps 3-7: Evaluate MSE for all candidate bandwidths m = m_min, ..., m_max
            m_candidates = range(m_min, m_max + 1)
            mse_values = {}

            if verbose:
                print(f"Evaluating bandwidths from {m_min} to {m_max}...")

            for m_eval in m_candidates:
                mse, _ = self._compute_bootstrap_mse(X, m_eval, d_current, k_n)
                mse_values[m_eval] = mse

            # Step 8: Choose \hat{m}_1 such that MSE*(\hat{m}_1) <= MSE*(m) for all m
            m_optimal = min(mse_values, key=mse_values.get)
            mse_optimal = mse_values[m_optimal]

            if verbose:
                print(f"Optimal bandwidth: {m_optimal}, MSE: {mse_optimal:.6f}")

            # Step 9: Check convergence criterion. Continue until
            # [MSE*(\hat{m}_l) - MSE*(\hat{m}_{l-1})] / MSE*(\hat{m}_{l-1}) > delta
            if len(mse_history) > 0:
                mse_prev = mse_history[-1]

                # Only check if both MSE values are finite and previous is non-zero
                if np.isfinite(mse_prev) and np.isfinite(mse_optimal) and mse_prev != 0:
                    relative_change = (mse_optimal - mse_prev) / mse_prev

                    if relative_change > self.delta:  # delta is negative
                        converged = True
                        # Step 9: Final bandwidth is \hat{m}_{l-1} (previous m)
                        m_optimal = m_current
                        if verbose:
                            print(f"Converged! Relative change: {relative_change:.4f}")
                        break

            mse_history.append(mse_optimal)
            # Replace m_1 with \hat{m}_1 and iterate
            m_current = m_optimal

        # Final fit with optimal bandwidth
        self.lw_estimator.fit(X, m=m_current)

        # Store fitted attributes
        self.optimal_m_ = m_current
        self.d_hat_ = self.lw_estimator.d_hat_
        self.se_ = self.lw_estimator.se_
        self.ase_ = self.lw_estimator.ase_
        self.mse_profile_ = mse_values
        self.iterations_ = iteration + 1
        self.converged_ = converged
        self.n_ = n
        self.objective_ = self.lw_estimator.objective_
        self.method_ = 'lw_bootstrap_m'

        return self

    def __repr__(self):
        """Representation showing key parameters."""
        params = []

        if self.k_n != 'auto':
            params.append(f"k_n={self.k_n}")
        if self.B != 200:
            params.append(f"B={self.B}")
        if self.delta != -0.01:
            params.append(f"delta={self.delta}")

        params_str = ", ".join(params) if params else ""
        return f"LWBootstrapM({params_str})"

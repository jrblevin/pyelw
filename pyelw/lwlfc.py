import numpy as np
from scipy.optimize import minimize
from typing import Optional, Dict, Any, Tuple


class LWLFC:
    """
    Modified Local Whittle estimator for low frequency contaminations.

    Implements the LWLFC (Local Whittle for Low Frequency Contamination)
    estimator of Hou and Perron (2014), which provides consistent estimation
    of the memory parameter d in the presence of low frequency contaminations
    such as random level shifts, deterministic level shifts, and trends.
    The estimator modifies the standard local Whittle objective by adding
    a term to account for the spectral behavior of LFC processes.

    Parameters
    ----------
    bounds : tuple of float, default=(-1.0, 2.2)
        Lower and upper bounds for optimization of memory parameter d.
    noise : bool, default=False
        If True, use the LWPLFC variant which additionally accounts for
        additive noise by including a constant term in the pseudo spectral
        density. This is recommended when additive noise contamination
        is suspected in addition to low frequency contamination.

    Attributes
    ----------
    d_hat_ : float
        Estimated memory parameter.
    theta_ : float
        Estimated signal-to-noise ratio for LFC component (theta_u for LWPLFC).
    theta_noise_ : float
        Estimated noise parameter (only for LWPLFC when noise=True).
    se_ : float
        Standard error of the estimate.
    ase_ : float
        Asymptotic standard error, equal to 1/(2*sqrt(m)).
    n_ : int
        Sample size.
    m_ : int
        Number of frequencies used.
    objective_ : float
        Final objective function value.
    nfev_ : int
        Number of function evaluations.
    method_ : str
        Estimation method used ('lwlfc' or 'lwplfc').

    Notes
    -----
    The pseudo spectral density used for LWLFC is:

        f_k = G_0 * (lambda_k^{-2d} + theta * lambda_k^{-2} / n)

    For LWPLFC (noise=True), an additional constant term is included:

        f_k = G_0 * (lambda_k^{-2d} + theta_w + theta_u * lambda_k^{-2} / n)

    The estimator jointly optimizes over d and the auxiliary parameter(s).
    The auxiliary parameters are not of primary interest but control the
    influence of contaminations at low frequencies.

    Bandwidth Selection
    -------------------
    Hou and Perron (2014) recommend m = n^alpha with alpha in {0.6, 0.7, 0.8}.
    A larger bandwidth (alpha=0.8) is preferable when only LFC is present.
    A smaller bandwidth (alpha=0.6) is better when short-memory dynamics
    are also present.

    References
    ----------
    Hou, J. and Perron, P. (2014). Modified local Whittle estimator for long
    memory processes in the presence of low frequency (and other)
    contaminations. _Journal of Econometrics_ 182, 309--328.
    """

    def __init__(self, bounds=(-1.0, 2.2), noise=False):
        self._default_bounds = (-1.0, 2.2)
        self._default_noise = False

        self.bounds = bounds
        self.noise = noise

    def prepare_data(self, X: np.ndarray, m: int) -> Dict[str, np.ndarray]:
        """
        Precompute quantities used for LWLFC estimation.

        Parameters
        ----------
        X : np.ndarray
            Time series data
        m : int
            Number of frequencies to use in estimation

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'n': sample size
            - 'm': number of frequencies
            - 'I_X': periodogram values at frequencies 1, ..., m
            - 'freqs': Fourier frequencies lambda_j = 2*pi*j/n
        """
        n = len(X)

        # Compute FFT and periodogram
        fft_X = np.fft.fft(X)
        I_X = np.abs(fft_X[1:m+1])**2 / (2 * np.pi * n)

        # Frequencies: lambda_j = 2*pi*j/n for j = 1, ..., m
        freqs = 2 * np.pi * np.arange(1, m+1, dtype=np.float64) / n

        return {
            'n': n,
            'm': m,
            'I_X': I_X,
            'freqs': freqs,
        }

    def objective(self, params: np.ndarray, data: Dict[str, np.ndarray]) -> float:
        """
        LWLFC objective function of Hou and Perron (2014, p. 312).

        Implements the concentrated quasi-maximum likelihood objective:

            J_m(d, theta) = log(1/m * sum_{k=1}^m I_k/g_k) + 1/m * sum_{k=1}^m log(g_k)

        where g_k = lambda_k^{-2d} + theta * lambda_k^{-2} / T.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [d, theta] for LWLFC or [d, theta_w, theta_u] for LWPLFC.
        data : Dict[str, np.ndarray]
            Precomputed quantities from prepare_data.

        Returns
        -------
        float
            Objective function value to be minimized.
        """
        freqs = data['freqs']
        I_X = data['I_X']
        n = data['n']

        try:
            if self.noise:
                # LWPLFC: g_k = lambda_k^{-2d} + theta_w + theta_u * lambda_k^{-2} / n
                d, theta_w, theta_u = params
                g_k = freqs**(-2*d) + theta_w + (theta_u / n) * freqs**(-2)
            else:
                # LWLFC: g_k = lambda_k^{-2d} + theta * lambda_k^{-2} / n
                d, theta = params
                g_k = freqs**(-2*d) + (theta / n) * freqs**(-2)

            # Check for valid g_k values
            if np.any(g_k <= 0):
                return np.float64(np.inf)

            # Objective function: log(mean(I_k/g_k)) + mean(log(g_k))
            mean_ratio = np.mean(I_X / g_k)
            if mean_ratio <= 0:
                return np.float64(np.inf)
            obj = np.log(mean_ratio) + np.mean(np.log(g_k))

            if not np.isfinite(obj):
                return np.float64(np.inf)

            return np.float64(obj)

        except (OverflowError, ZeroDivisionError, ValueError):
            return np.float64(np.inf)

    def fit(self, X, m=None, verbose=False):
        """
        LWLFC estimation of memory parameter d.

        Parameters
        ----------
        X : np.ndarray
            Time series data.
        m : int, optional
            Number of frequencies to use. If None, uses n^0.8 as recommended
            by Hou and Perron (2014) for the case with only LFC.
        verbose : bool, default=False
            Print diagnostic information during fitting.

        Returns
        -------
        self : object
            Returns the fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64).flatten()
        n = len(X)

        # Default bandwidth
        if m is None:
            m = int(n**0.8)

        # Prepare data
        data = self.prepare_data(X, m)

        # Set up optimization bounds.
        # Matches the R LongMemoryTS Hou.Perron implementation.
        if self.noise:
            # LWPLFC: [d, theta_w, theta_u]
            # theta_w >= 0 (noise variance ratio)
            # theta_u >= 0 (LFC signal-to-noise ratio)
            param_bounds = [
                self.bounds,      # d bounds
                (0.0, 10000),     # theta_w in [0, 10000]
                (0.0, 10000),     # theta_u in [0, 10000]
            ]
            method = 'lwplfc'
        else:
            # LWLFC: [d, theta]
            # theta in [0, 10000] (signal-to-noise ratio, matching R)
            param_bounds = [
                self.bounds,      # d bounds
                (0.0, 10000),     # theta in [0, 10000]
            ]
            method = 'lwlfc'

        # Objective function wrapper
        nfev = [0]

        def obj_func(params):
            nfev[0] += 1
            return self.objective(params, data)

        # Starting point: (d=0, theta=0).
        # Matches the R LongMemoryTS Hou.Perron implementation.
        if self.noise:
            x0 = np.array([0.0, 0.0, 0.0])
        else:
            x0 = np.array([0.0, 0.0])

        # L-BFGS-B optimization.
        # Matches the R LongMemoryTS Hou.Perron implementation.
        try:
            best_result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=param_bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
        except Exception:
            best_result = None

        if best_result is None or not np.isfinite(best_result.fun):
            d_hat = np.nan
            theta = np.nan
            theta_noise = np.nan
            final_obj = np.nan
        else:
            d_hat = best_result.x[0]
            if self.noise:
                theta_noise = best_result.x[1]
                theta = best_result.x[2]
            else:
                theta = best_result.x[1]
                theta_noise = np.nan
            final_obj = best_result.fun

        # Asymptotic standard error: Theorem 2 of Hou and Perron (2014)
        ase = 1 / (2 * np.sqrt(m))

        # Finite-sample standard error via numerical Hessian
        if np.isfinite(d_hat) and best_result is not None:
            try:
                # Compute numerical Hessian at the optimum
                eps = 1e-5
                if self.noise:
                    params_opt = np.array([d_hat, theta_noise, theta])
                else:
                    params_opt = np.array([d_hat, theta])

                # Second derivative w.r.t. d (first parameter)
                d_plus = params_opt.copy()
                d_plus[0] += eps
                d_minus = params_opt.copy()
                d_minus[0] -= eps

                f_plus = self.objective(d_plus, data)
                f_minus = self.objective(d_minus, data)
                f_center = final_obj

                d2_dd = (f_plus - 2*f_center + f_minus) / (eps**2)

                if d2_dd > 0:
                    se = np.sqrt(1 / (m * d2_dd))
                else:
                    se = ase  # Fall back to asymptotic SE
            except Exception:
                se = ase
        else:
            se = np.nan

        # Store fitted attributes
        self.n_ = n
        self.m_ = m
        self.d_hat_ = d_hat
        self.theta_ = theta
        self.theta_noise_ = theta_noise if self.noise else np.nan
        self.se_ = se
        self.ase_ = ase
        self.objective_ = final_obj
        self.nfev_ = nfev[0]
        self.method_ = method

        if verbose:
            print("LWLFC Estimation Results:")
            print(f"  d_hat = {d_hat:.6f}")
            print(f"  theta = {theta:.6f}")
            if self.noise:
                print(f"  theta_noise = {theta_noise:.6f}")
            print(f"  se = {se:.6f}")
            print(f"  ase = {ase:.6f}")
            print(f"  n = {n}, m = {m}")
            print(f"  nfev = {nfev[0]}")

        return self

    def estimate(self,
                 X: np.ndarray,
                 m=None,
                 bounds: Optional[Tuple[float, float]] = None,
                 noise: Optional[bool] = None,
                 verbose: Optional[bool] = False) -> Dict[str, Any]:
        """
        LWLFC estimation of memory parameter d.

        This method provides backward compatibility with the original API.
        For new code, use fit() and access fitted attributes directly.

        Parameters
        ----------
        X : np.ndarray
            Time series data.
        m : int, optional
            Number of frequencies to use.
        bounds : tuple of float, optional
            Lower and upper bounds for optimization.
            If provided, temporarily overrides constructor bounds.
        noise : bool, optional
            If True, use LWPLFC variant.
            If provided, temporarily overrides constructor noise setting.
        verbose : bool, optional
            Print diagnostic information.

        Returns
        -------
        Dict[str, Any]
            Dictionary with estimation results.
        """
        # Temporarily store original parameters
        original_bounds = self.bounds
        original_noise = self.noise

        # Override parameters if provided
        if bounds is not None:
            self.bounds = bounds
        if noise is not None:
            self.noise = noise

        try:
            # Fit the model
            self.fit(X, m=m, verbose=verbose)

            # Return results as dictionary
            result = {
                'n': self.n_,
                'm': self.m_,
                'd_hat': self.d_hat_,
                'theta': self.theta_,
                'se': self.se_,
                'ase': self.ase_,
                'objective': self.objective_,
                'nfev': self.nfev_,
                'method': self.method_,
            }
            if self.noise:
                result['theta_noise'] = self.theta_noise_

            return result
        finally:
            # Restore original parameters
            self.bounds = original_bounds
            self.noise = original_noise

    def __repr__(self):
        """Representation showing non-default parameters."""
        params = []

        if self.bounds != self._default_bounds:
            params.append(f"bounds={self.bounds}")
        if self.noise != self._default_noise:
            params.append(f"noise={self.noise}")

        params_str = ", ".join(params)
        return f"LWLFC({params_str})"

    def __str__(self):
        return self.__repr__()

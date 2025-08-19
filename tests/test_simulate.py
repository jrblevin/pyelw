import pytest  # noqa: F401
import numpy as np
from math import gamma

from pyelw.simulate import arfima


def test_basic_properties():
    """Test basic properties of ARFIMA simulation."""
    n = 10000
    d = 0.2
    seed = 42
    sigma = 1.0
    x = arfima(n, d, sigma=sigma, seed=seed)

    assert len(x) == n
    assert np.isfinite(x).all()
    assert abs(np.mean(x)) < 0.01  # Approximately zero mean

    # Check theoretical variance for stationary case
    var_true = sigma**2 * gamma(1 - 2*d) / (gamma(1 - d)**2)
    var_est = np.var(x)
    relative_error = abs(var_est - var_true) / var_true
    assert relative_error < 0.01, f"Variance error: {relative_error:.3f}"


def test_sigma_scaling():
    """Test that sigma parameter scales variance correctly."""
    n = 10000
    d = 0.3
    seed = 42
    sigma1 = 1.0
    sigma2 = 2.0
    x1 = arfima(n, d, sigma=sigma1, seed=seed)
    x2 = arfima(n, d, sigma=sigma2, seed=seed)

    # Variance should scale as sigma^2
    ratio = np.var(x2) / np.var(x1)
    assert 3.9 < ratio < 4.1  # Approximately 4

    # Check theoretical variance for stationary cases
    var1_true = sigma1**2 * gamma(1 - 2*d) / (gamma(1 - d)**2)
    var1_est = np.var(x1)
    relative_error = abs(var1_est - var1_true) / var1_true
    assert relative_error < 0.02, f"Variance error (sigma = 1): {relative_error:.3f}"

    var2_true = sigma2**2 * gamma(1 - 2*d) / (gamma(1 - d)**2)
    var2_est = np.var(x2)
    relative_error = abs(var2_est - var2_true) / var2_true
    assert relative_error < 0.02, f"Variance error (sigma = 2): {relative_error:.3f}"


def test_extreme_d_values():
    """Test ARFIMA with extreme but valid d values."""
    n = 500
    seed = 42

    # Test near-boundaries and extreme values
    extreme_d = [-0.49, 0.49, 1.5, 2.0]
    for d in extreme_d:
        x = arfima(n, d, seed=seed)
        assert len(x) == n
        assert np.isfinite(x).all()


def test_white_noise():
    """Test that d=0 produces approximately white noise."""
    n = 10000
    d = 0.0
    seed = 42
    x = arfima(n, d, seed=seed)

    # Should be approximately white noise
    assert abs(np.mean(x)) < 0.01
    assert 0.99 < np.std(x) < 1.01

    # Autocorrelations should be small
    x_centered = x - np.mean(x)
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    assert abs(autocorr[1]) < 0.02
    assert abs(autocorr[2]) < 0.02
    assert abs(autocorr[3]) < 0.02


@pytest.mark.parametrize("d", [0.3, 0.4, 0.45, 0.6, 0.9, 1.0, 1.2, 1.4])
def test_memory_and_persistence_properties(d):
    """Test memory properties (stationary d < 0.5) and persistence (nonstationary d ≥ 0.5)."""
    n = 5000
    seed = 42
    x = arfima(n, d, seed=seed)

    x_centered = x - np.mean(x)
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]  # Normalize

    if d < 0.5:
        # Test stationary long memory: positive autocorrelations at medium lags
        assert autocorr[10] > 0.1, f"Stationary long memory d={d}: rho(10) should show persistence"
        assert autocorr[20] > 0.05, f"Stationary long memory d={d}: rho(20) should remain positive"
    else:
        # Test strong persistence due to nonstationarity
        assert autocorr[20] > 0.5, f"Nonstationary persistence d={d}: rho(20) should be > 0.5"


@pytest.mark.parametrize("d", [-0.4, -0.2, 0.0, 0.2, 0.4, 0.45])
def test_spectral_density_behavior(d):
    """Test behavior of spectral density in stationary cases."""
    n = 10000
    seed = 42
    x = arfima(n, d, seed=seed)

    # Compute periodogram
    fft_x = np.fft.fft(x)
    periodogram = np.abs(fft_x)**2 / n
    freqs = 2 * np.pi * np.arange(n) / n
    expected_slope = -2 * d

    # Use proportional frequency range (1% of frequencies)
    num_freqs = int(0.01 * n)
    test_freqs = freqs[1:num_freqs+1]
    test_periodo = periodogram[1:num_freqs+1]

    # Fit slope in log-log space
    log_periodo = np.log(test_periodo)
    log_freqs = np.log(test_freqs)
    slope = np.polyfit(log_freqs, log_periodo, 1)[0]

    # Test with realistic tolerance
    error = abs(slope - expected_slope)
    assert error < 0.15, f"Spectral slope error {error:.3f} > 0.15 (slope={slope:.3f}, expected={expected_slope:.3f})"


@pytest.mark.parametrize("d", [0.5, 0.6, 0.9, 1.0, 1.2, 1.4])
def test_variance_growth_nonstationary(d):
    """Test that variance grows with n for nonstationary d >= 0.5."""
    seed = 42

    # Test different sample sizes
    var_250 = np.var(arfima(250, d, seed=seed))
    var_500 = np.var(arfima(500, d, seed=seed))
    var_1000 = np.var(arfima(1000, d, seed=seed))

    # Variance should grow with sample size for d >= 0.5
    assert var_500 > var_250
    assert var_1000 > var_500


def test_stationarity_boundary():
    """Test behavior around d = 0.5 stationarity boundary."""
    n = 1000
    seed = 42

    # Stationary case: d < 0.5
    x_stat = arfima(n, 0.45, seed=seed)
    var_stat = np.var(x_stat)

    # Nonstationary case: d > 0.5
    x_nonstat = arfima(n, 0.55, seed=seed)
    var_nonstat = np.var(x_nonstat)

    # Nonstationary should have higher variance
    assert var_nonstat > var_stat


@pytest.mark.parametrize("d", [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.45])
def test_unconditional_moments_stationary(d):
    """Test first four unconditional moments for stationary case."""
    n = 10000
    sigma = 1.0
    seed = 42
    x = arfima(n, d=d, sigma=sigma, seed=seed)

    # First moment (mean) should be 0
    mean_emp = np.mean(x)
    assert abs(mean_emp) < 0.15, f"Mean should be ~0, got {mean_emp:.4f}"

    # Second moment (variance), use theoretical formula
    # For stationary ARFIMA(0,d,0) with u_t ~ N(0,\sigma^2):
    # Var(X_t) = \sigma^2 * \Gamma(1-2d) / [\Gamma(1-d)^2] for d < 0.5
    var_true = sigma**2 * gamma(1 - 2*d) / (gamma(1 - d)**2)
    var_est = np.var(x)
    relative_error = abs(var_est - var_true) / var_true
    assert relative_error < 0.5, f"Variance error: {relative_error:.3f}"

    # Third moment (skewness) should be 0, symmetric distribution
    from scipy.stats import skew
    skewness = skew(x, bias=False)
    assert abs(skewness) < 0.2, f"Skewness should be ~0, got {skewness:.4f}"

    # Fourth moment (excess kurtosis) should be 0 for Gaussian innovations
    from scipy.stats import kurtosis
    excess_kurt = kurtosis(x, bias=False, fisher=True)
    assert abs(excess_kurt) < 0.5, f"Excess kurtosis should be ~0, got {excess_kurt:.4f}"


@pytest.mark.parametrize("d", [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
def test_autocorrelation_stationary(d):
    """
    Comprehensive test of ARFIMA(0,d,0) autocorrelation function.
    """
    n = 20000
    seed = 42
    x = arfima(n, d, seed=seed)
    x_centered = x - np.mean(x)

    # Compute empirical autocorrelations
    ac_est = np.correlate(x_centered, x_centered, mode='full')
    ac_est = ac_est[len(ac_est)//2:]
    ac_est = ac_est / ac_est[0]

    # Part 1: Test Exact Autocorrelation Values
    # -----------------------------------------

    # Compute theoretical autocorrelations for lags 1-10
    short_lags = np.arange(1, 11)
    if abs(d) < 1e-10:  # d ≈ 0, white noise case
        ac_true_short = np.zeros(len(short_lags))
    else:
        ac_true_short = np.zeros(len(short_lags))
        for i, k in enumerate(short_lags):
            ac_true_short[i] = (gamma(k + d) * gamma(1 - d)) / (gamma(k + 1 - d) * gamma(d))

    # Adaptive tolerance based on d value
    if abs(d) < 0.15:
        rel_tol = 0.30  # Higher tolerance for small |d|
    else:
        rel_tol = 0.20

    # Compare empirical vs theoretical for short lags
    for i, k in enumerate(short_lags):
        if abs(ac_true_short[i]) < 0.05:  # Small theoretical value - use absolute error
            error = abs(ac_est[k] - ac_true_short[i])
            assert error < 0.05, f"d={d}, Lag {k}: abs error={error:.3f} > 0.05"
        else:  # Large theoretical value - use relative error
            relative_error = abs(ac_est[k] - ac_true_short[i]) / abs(ac_true_short[i])
            assert relative_error < rel_tol, f"d={d}, Lag {k}: rel error={relative_error:.3f} > {rel_tol}"

    # Part 2: Special Tests Based on d
    # --------------------------------

    if abs(d) < 0.05:  # Approximately white noise
        # All autocorrelations should be small
        max_ac = np.max(np.abs(ac_est[1:11]))
        expected_se = 1.0 / np.sqrt(n)  # Standard error for white noise
        assert max_ac < 4 * expected_se, \
            f"Near white noise d={d}: max |rho(k)| = {max_ac:.4f} > {4*expected_se:.4f}"

    elif d < -0.05:
        # Negative d: First autocorrelation should be negative
        assert ac_est[1] < 0, f"Antipersistent d={d}: rho(1) = {ac_est[1]:.3f} should be negative"

    # Part 3: Test Asymptotic Decay (Long Lags)
    # -----------------------------------------

    # Only test when d > 0.05 to exclude white noise and negative d where
    # alternating signs complicate tests
    if d < 0.05:
        return

    # For ARFIMA(0,d,0), |\rho(k)| ~ k^(2d-1) as k \to \infty
    long_lags = np.array([10, 15, 20, 25, 30])

    # Test magnitude decay: |\rho(k2)|/|\rho(k1)| ≈ (k1/k2)^(1-2d)
    magnitude_ratios = []
    theoretical_ratios = []
    for i in range(len(long_lags) - 1):
        k1, k2 = long_lags[i], long_lags[i + 1]

        # Use magnitudes to avoid sign issues
        mag1, mag2 = abs(ac_est[k1]), abs(ac_est[k2])

        # Skip if autocorrelations are too small (noise dominates)
        if mag1 > 0.01 and mag2 > 0.005:
            emp_ratio = mag2 / mag1
            theo_ratio = (k1 / k2) ** (1 - 2*d)

            magnitude_ratios.append(emp_ratio)
            theoretical_ratios.append(theo_ratio)

    # Test magnitude decay ratios
    if magnitude_ratios:
        for emp, theo in zip(magnitude_ratios, theoretical_ratios):
            ratio_of_ratios = emp / theo if theo > 1e-10 else float('inf')
            # More lenient bounds for asymptotic behavior
            assert 0.3 < ratio_of_ratios < 3.0, \
                f"d={d}, Asymptotic decay: magnitude ratio = {ratio_of_ratios:.3f} not in [0.3, 3.0]"


def test_long_run_variance_scaling():
    """Test long-run variance scaling for different stationary d values."""
    n = 2000
    sigma = 1.0
    seed = 42
    d_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45]
    variances = []
    for d in d_values:
        x = arfima(n, d, sigma=sigma, seed=seed)
        variances.append(np.var(x))

    # Variance should increase monotonically in d
    for i in range(1, len(variances)):
        assert variances[i] > variances[i-1], \
            f"Variance not increasing: d={d_values[i-1]} -> {d_values[i]}"


@pytest.mark.parametrize("d", [-0.3, -0.2, -0.1])
def test_negative_d_antipersistence(d):
    """Test that negative d produces antipersistent behavior."""
    n = 1000
    seed = 42
    x = arfima(n, d, seed=seed)

    # Compute autocorrelations
    x_centered = x - np.mean(x)
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # For negative d, first autocorrelation should be negative
    assert autocorr[1] < 0, f"Antipersistent d={d}: rho(1) = {autocorr[1]:.3f} should be negative"


@pytest.mark.parametrize("phi", [0.3, 0.5, 0.7, 0.9, -0.3, -0.5, -0.7])
def test_pure_ar1_behavior(phi):
    """Test pure AR(1) process (d=0, phi!=0)."""
    n = 10000
    seed = 42
    sigma = 1.0
    x = arfima(n, d=0.0, phi=phi, sigma=sigma, seed=seed)

    # Test 1: Mean should be approximately 0
    assert abs(np.mean(x)) < 0.025, f"AR(1) phi={phi}: mean = {np.mean(x):.3f}"

    # Test 2: Variance should match theoretical value \sigma^2/(1-\phi^2)
    var = sigma**2 / (1 - phi**2)
    var_est = np.var(x)
    rel_err = abs(var_est - var) / var
    # Higher tolerance for near-unit-root cases
    tol = 0.1 if abs(phi) > 0.8 else 0.05
    assert rel_err < tol, f"AR(1) phi={phi}: variance error = {rel_err:.3f}"

    # Test 3: Autocorrelation function should be \rho(k) = \phi^k
    x_centered = x - np.mean(x)
    acorr = np.correlate(x_centered, x_centered, mode='full')
    acorr = acorr[len(acorr)//2:]
    acorr = acorr / acorr[0]

    # Check first 5 autocorrelations
    for k in range(1, 6):
        rho = phi**k
        rho_est = acorr[k]
        error = abs(rho_est - rho)
        assert error < 0.03, f"AR(1) phi={phi}, lag {k}: |rho({k}) - phi^{k}| = {error:.3f}"


@pytest.mark.parametrize("phi", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("d", [0.1, 0.2, 0.3, 0.4])
def test_arfima_acorr(phi, d):
    """Test ARFIMA(1,d,0) autocorrelations."""
    n = 5000
    seed = 42
    sigma = 1.0
    x = arfima(n, d=d, phi=phi, sigma=sigma, seed=seed)

    # Basic checks
    assert len(x) == n
    assert np.isfinite(x).all()

    # Calculate autocorrelations
    x_centered = x - np.mean(x)
    acorr = np.correlate(x_centered, x_centered, mode='full')
    acorr = acorr[len(acorr)//2:]
    acorr = acorr / acorr[0]

    # For ARFIMA(0,d,0), \rho(1) = \Gamma(1+d)\Gamma(1-d)/(\Gamma(2-d)\Gamma(d))
    rho1_arfima_0d0 = (gamma(1 + d) * gamma(1 - d)) / (gamma(2 - d) * gamma(d))

    # For ARFIMA(1,d,0) with \phi > 0, \rho(1) should be larger than pure fractional case
    if phi > 0:
        assert acorr[1] > rho1_arfima_0d0, \
            f"ARFIMA(1,d,0) phi={phi}, d={d}: rho(1)={acorr[1]:.3f} should be > {rho1_arfima_0d0:.3f}"


def test_spectral_density_with_ar():
    """Test spectral density for ARFIMA(1,d,0) process."""
    n = 10000
    seed = 42
    sigma = 1.0

    # Test cases with different phi and d combinations
    test_cases = [
        (0.5, 0.0),   # Pure AR(1)
        (0.0, 0.3),   # Pure ARFIMA(0,d,0)
        (0.5, 0.2),   # ARFIMA(1,d,0) with positive phi
        (-0.4, 0.2),  # ARFIMA(1,d,0) with negative phi
    ]

    for phi, d in test_cases:
        x = arfima(n, d=d, phi=phi, sigma=sigma, seed=seed)

        # Compute periodogram
        fft_x = np.fft.fft(x - np.mean(x))
        periodogram = np.abs(fft_x)**2 / n
        freqs = 2 * np.pi * np.arange(n) / n

        # For pure AR(1) (d=0), spectral density is:
        # f(\lambda) = \sigma^2/(2\pi) * 1/|1 - \phi e^(-i\lambda)|^2
        #            = \sigma^2/(2\pi) * 1/(1 - 2\phi\cos(\lambda) + \phi^2)
        if abs(d) < 1e-10 and abs(phi) > 1e-10:
            # Test at a few frequencies
            test_freqs_idx = [10, 50, 100, 200]
            for idx in test_freqs_idx:
                lambda_freq = freqs[idx]
                theoretical_sd = (sigma**2 / (2 * np.pi)) / (1 - 2*phi*np.cos(lambda_freq) + phi**2)
                empirical_sd = periodogram[idx]

                # Periodogram is noisy, so use log scale and allow larger tolerance
                if theoretical_sd > 0.01 and empirical_sd > 0.01:
                    log_ratio = np.log(empirical_sd / theoretical_sd)
                    assert abs(log_ratio) < 3.0, \
                        f"AR(1) phi={phi}, freq idx={idx}: log ratio = {log_ratio:.3f}"

        # For ARFIMA(1,d,0), just check that spectrum is well-behaved
        assert np.isfinite(periodogram).all(), f"ARFIMA({phi:.1f},{d:.1f},0): non-finite spectrum"
        assert (periodogram[1:n//2] > 0).all(), f"ARFIMA({phi:.1f},{d:.1f},0): non-positive spectrum"


def test_ar1_initialization():
    """Test correct initialization of AR(1) component."""
    seed = 42
    sigma = 1.0
    n_short = 100  # Short series to test initial conditions effect
    n_long = 10000  # Long series for steady-state properties

    # Test 1: Stationary AR(1) initialization
    phi_stationary = [0.3, 0.5, 0.7, 0.9, -0.5]

    for phi in phi_stationary:
        # Generate many short series to test initial value distribution
        initial_values = []
        for i in range(1000):
            x = arfima(n_short, d=0.0, phi=phi, sigma=sigma, seed=seed+i)
            initial_values.append(x[0])

        # Check initial value distribution
        # Should be approximately N(0, \sigma^2/(1-\phi^2))
        init_var_theoretical = sigma**2 / (1 - phi**2)
        init_var_empirical = np.var(initial_values)
        relative_error = abs(init_var_empirical - init_var_theoretical) / init_var_theoretical

        # More lenient for this test since we're looking at first values
        assert relative_error < 0.2, \
            f"AR(1) initialization phi={phi}: variance error = {relative_error:.3f}"

    # Test 2: Check that process reaches steady state quickly
    phi = 0.7
    x = arfima(n_long, d=0.0, phi=phi, sigma=sigma, seed=seed)

    # Compare variance of first quarter vs last quarter
    quarter_size = n_long // 4
    var_first = np.var(x[:quarter_size])
    var_last = np.var(x[-quarter_size:])

    # Both should be close to theoretical variance
    var_theoretical = sigma**2 / (1 - phi**2)
    assert abs(var_first - var_theoretical) / var_theoretical < 0.1, \
        f"First quarter variance off by {abs(var_first - var_theoretical) / var_theoretical:.3f}"
    assert abs(var_last - var_theoretical) / var_theoretical < 0.1, \
        f"Last quarter variance off by {abs(var_last - var_theoretical) / var_theoretical:.3f}"

    # Test 3: Unit root case (phi = 1) should be handled
    x_unit = arfima(1000, d=0.0, phi=1.0, sigma=sigma, seed=seed)
    assert np.isfinite(x_unit).all(), "Unit root AR(1) produced non-finite values"

    # For unit root, variance should grow over time
    # Use cumulative variance to see the growth pattern
    var_early = np.var(x_unit[:100])
    var_late = np.var(x_unit[:900])  # Include more data to see growth
    assert var_late > var_early, f"Unit root variance should increase: early={var_early:.2f}, late={var_late:.2f}"

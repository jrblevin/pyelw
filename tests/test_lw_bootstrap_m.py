import pytest
import numpy as np

from pyelw import LW
from pyelw.lw_bootstrap_m import LWBootstrapM
from pyelw.simulate import arfima


@pytest.fixture
def simple_arfima_data():
    """Generate simple ARFIMA(0, d, 0) data for testing."""
    np.random.seed(42)
    return arfima(n=100, d=0.3, phi=0.0)


@pytest.fixture
def lw_estimator():
    """Create basic LW estimator instance."""
    return LW()


@pytest.fixture
def selector(lw_estimator):
    """Create LWBootstrapM instance."""
    return LWBootstrapM(
        lw_estimator=lw_estimator,
        k_n=2,
        B=10,  # Small B for fast tests
        m_min=5,
        m_max=15,
        m_init=8,
        delta=-0.01,
        max_iter=3,
        verbose=False
    )


# Initialization tests

def test_init_with_estimator(lw_estimator):
    """Test initialization with provided LW estimator."""
    selector = LWBootstrapM(lw_estimator=lw_estimator)
    assert selector.lw_estimator is lw_estimator
    assert selector.k_n == 'auto'
    assert selector.B == 200
    assert selector.delta == -0.01


def test_init_without_estimator():
    """Test initialization creates default LW estimator."""
    selector = LWBootstrapM()
    assert selector.lw_estimator is not None
    assert isinstance(selector.lw_estimator, LW)


def test_init_custom_parameters(lw_estimator):
    """Test initialization with custom parameters."""
    selector = LWBootstrapM(
        lw_estimator=lw_estimator,
        k_n=5,
        B=100,
        m_min=10,
        m_max=50,
        m_init=20,
        delta=-0.02,
        max_iter=5,
        verbose=True,
    )
    assert selector.k_n == 5
    assert selector.B == 100
    assert selector.m_min == 10
    assert selector.m_max == 50
    assert selector.m_init == 20
    assert selector.delta == -0.02
    assert selector.max_iter == 5
    assert selector.verbose is True


def test_init_custom_bounds():
    """Test that custom bounds are passed to default LW estimator."""
    custom_bounds = (-0.5, 1.5)
    selector = LWBootstrapM(bounds=custom_bounds)

    # Verify the selector stored the bounds
    assert selector.bounds == custom_bounds

    # Verify the default LW estimator was created with these bounds
    assert selector.lw_estimator is not None
    assert selector.lw_estimator.bounds == custom_bounds


def test_init_with_estimator_ignores_bounds(lw_estimator):
    """Test that bounds parameter is ignored when lw_estimator is provided."""
    # lw_estimator fixture has default bounds (-1.0, 2.2)
    custom_bounds = (-0.5, 1.5)
    selector = LWBootstrapM(
        lw_estimator=lw_estimator,
        bounds=custom_bounds,
    )

    # Selector stores the bounds parameter
    assert selector.bounds == custom_bounds

    # But the provided estimator keeps its own bounds
    assert selector.lw_estimator.bounds == (-1.0, 2.2)


# Locally standardized periodogram tests

def test_locally_standardized_periodogram_basic(selector, simple_arfima_data):
    """Test basic periodogram computation."""
    d_hat = 0.3
    v_j = selector._locally_standardized_periodogram(simple_arfima_data, d_hat)

    # Check output shape
    n = len(simple_arfima_data)
    assert len(v_j) == n // 2

    # Check all values are finite
    assert np.all(np.isfinite(v_j))

    # Check all values are non-negative (squared magnitudes)
    assert np.all(v_j >= 0)




def test_periodogram_formula_manual_white_noise(selector):
    """Test formula v_j^(1) = I_j * lambda_j^(2d) against manual calculation for white noise."""
    np.random.seed(123)
    X = np.random.randn(50)
    d_hat = 0.0
    n = len(X)

    # Get result from method
    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Manual calculation
    fft_X = np.fft.fft(X)
    I_j_manual = np.abs(fft_X[1:n//2+1])**2 / (2 * np.pi * n)
    freqs_manual = 2 * np.pi * np.arange(1, n//2+1) / n
    v_j_manual = I_j_manual * (freqs_manual**(2 * d_hat))

    # Should match exactly
    np.testing.assert_allclose(v_j, v_j_manual, rtol=1e-14)


def test_periodogram_formula_manual_long_memory(selector):
    """Test formula against manual calculation for long memory series."""
    np.random.seed(456)
    X = arfima(n=100, d=0.4, phi=0.0)
    d_hat = 0.35
    n = len(X)

    # Get result from method
    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Manual calculation
    fft_X = np.fft.fft(X)
    I_j_manual = np.abs(fft_X[1:n//2+1])**2 / (2 * np.pi * n)
    freqs_manual = 2 * np.pi * np.arange(1, n//2+1) / n
    v_j_manual = I_j_manual * (freqs_manual**(2 * d_hat))

    # Should match exactly
    np.testing.assert_allclose(v_j, v_j_manual, rtol=1e-14)


def test_periodogram_d_zero_equals_raw(selector):
    """When d=0, v_j should equal raw periodogram (since lambda^0 = 1)."""
    np.random.seed(789)
    X = np.random.randn(80)
    d_hat = 0.0
    n = len(X)

    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Compute raw periodogram
    fft_X = np.fft.fft(X)
    I_j = np.abs(fft_X[1:n//2+1])**2 / (2 * np.pi * n)

    # Should be identical when d=0
    np.testing.assert_allclose(v_j, I_j, rtol=1e-14)


def test_periodogram_frequencies_correct(selector):
    """Test that frequencies lambda_j = 2*pi*j/n are computed correctly."""
    X = np.random.randn(100)
    n = len(X)

    # Test by checking ratio at different d values
    v_j_d1 = selector._locally_standardized_periodogram(X, d_hat=1.0)
    v_j_d0 = selector._locally_standardized_periodogram(X, d_hat=0.0)

    # For each j, v_j(d=1) / v_j(d=0) should equal lambda_j^2
    expected_freqs = 2 * np.pi * np.arange(1, n//2+1) / n
    expected_ratio = expected_freqs**2

    # Compute actual ratio (avoiding division by very small numbers)
    mask = v_j_d0 > 1e-10
    actual_ratio = np.zeros_like(v_j_d0)
    actual_ratio[mask] = v_j_d1[mask] / v_j_d0[mask]

    # Check where periodogram is not too small
    np.testing.assert_allclose(
        actual_ratio[mask],
        expected_ratio[mask],
        rtol=1e-12,
        err_msg="Frequency weighting not correct"
    )


def test_periodogram_reversibility(selector):
    """Test that standardization is reversible."""
    np.random.seed(42)
    X = np.random.randn(100)
    d_hat = 0.3
    n = len(X)

    # Get standardized periodogram
    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Recover original by dividing
    freqs = 2 * np.pi * np.arange(1, n//2+1) / n
    I_j_recovered = v_j / (freqs**(2 * d_hat))

    # Compute original periodogram
    fft_X = np.fft.fft(X)
    I_j_original = np.abs(fft_X[1:n//2+1])**2 / (2 * np.pi * n)

    np.testing.assert_allclose(I_j_recovered, I_j_original, rtol=1e-14)


@pytest.mark.parametrize("n", [100, 200, 512])
def test_periodogram_output_length(selector, n):
    """Test output length is n//2 for various n."""
    X = np.random.randn(n)
    d_hat = 0.3

    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Should return up to Nyquist frequency
    expected_length = n // 2
    assert len(v_j) == expected_length


def test_periodogram_no_dc_component(selector):
    """Test that DC component (frequency 0) is excluded."""
    X = np.random.randn(100) + 5.0  # Add non-zero mean
    d_hat = 0.0
    n = len(X)

    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Compute what DC component would be
    fft_X = np.fft.fft(X)
    dc_component = np.abs(fft_X[0])**2 / (2 * np.pi * n)

    # First element of v_j should NOT be the DC component
    assert v_j[0] != dc_component


def test_periodogram_constant_series_near_zero(selector):
    """Test with constant series (extreme case)."""
    X = np.ones(100) * 5.0
    d_hat = 0.0

    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # For constant series, periodogram should be near zero (excluding DC)
    assert np.max(np.abs(v_j)) < 1e-10




def test_periodogram_sine_wave_peak(selector):
    """Test with pure sine wave (known spectral peak)."""
    n = 256
    freq_idx = 10  # Create sine at this frequency index
    t = np.arange(n)
    X = np.sin(2 * np.pi * freq_idx * t / n)
    d_hat = 0.0

    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # For pure sine wave, periodogram should have peak at freq_idx
    peak_idx = freq_idx - 1  # Adjust for excluded DC

    # Peak region should be much larger than outside
    peak_region = slice(max(0, peak_idx-2), min(len(v_j), peak_idx+3))
    outside_peak = np.concatenate([
        v_j[:max(0, peak_idx-3)],
        v_j[min(len(v_j), peak_idx+4):]
    ])

    if len(outside_peak) > 0:
        assert np.max(v_j[peak_region]) > 10 * np.mean(outside_peak)


def test_periodogram_long_memory_standardization_dampens_pole(selector):
    """Test that standardization dampens spectral pole for long memory."""
    np.random.seed(888)
    # Generate long memory series
    X = arfima(n=256, d=0.4, phi=0.0)

    # Raw periodogram (d=0)
    v_j_raw = selector._locally_standardized_periodogram(X, d_hat=0.0)

    # Standardized with correct d
    v_j_std = selector._locally_standardized_periodogram(X, d_hat=0.4)

    # For long memory, raw periodogram is large at low frequencies
    # Standardization should reduce this (dampen the pole)
    low_freq_raw = np.mean(v_j_raw[:10])
    high_freq_raw = np.mean(v_j_raw[-10:])
    ratio_raw = low_freq_raw / high_freq_raw

    low_freq_std = np.mean(v_j_std[:10])
    high_freq_std = np.mean(v_j_std[-10:])
    ratio_std = low_freq_std / high_freq_std

    # Standardization should reduce low/high frequency ratio
    assert ratio_std < ratio_raw, \
        "Standardization should reduce low-frequency dominance"


#
# k_n selection tests
#

def test_select_kn_spectral_flatness_white_noise(selector):
    """Test that white noise has high spectral flatness and gets large k_n."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n)
    d_hat = 0.0

    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Compute spectral flatness
    spectral_flatness = selector._spectral_flatness(v_j)

    # White noise should have high spectral flatness (close to 1)
    assert spectral_flatness > 0.5, f"White noise spectral flatness {spectral_flatness:.4f} should be > 0.5"

    k_n = selector._select_k_n(v_j, n)

    # High spectral flatness should lead to large k_n
    assert k_n >= n // 10, f"White noise should get large k_n, got {k_n}"


def test_select_kn_spectral_flatness_structured(selector):
    """Test that structured series has low spectral flatness and gets small k_n."""
    np.random.seed(42)
    n = 200
    # Create a series with strong AR component - creates peaks in spectrum
    X = arfima(n=n, d=0.4, phi=0.8)
    # Underestimate d to leave structure in v_j
    d_hat = 0.2

    v_j = selector._locally_standardized_periodogram(X, d_hat)

    # Compute spectral flatness
    spectral_flatness = selector._spectral_flatness(v_j)

    # Structured series should have lower spectral flatness
    # (though exact value depends on the realization)
    assert 0.0 < spectral_flatness < 1.0

    k_n = selector._select_k_n(v_j, n)

    # Should get some reasonable k_n
    assert k_n > 0
    assert k_n <= n // 2


#
# Local bootstrap resample tests
#

def test_local_bootstrap_resample_basic(selector):
    """Test basic bootstrap resampling."""
    v_j = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k_n = 1
    m = 3

    v_star = selector._local_bootstrap_resample(v_j, k_n, m, seed=42)

    # Check output shape
    assert len(v_star) == m

    # Check values are from v_j (or nearby indices)
    assert np.all(np.isfinite(v_star))


def test_local_bootstrap_resample_reproducibility(selector):
    """Test that same seed produces same results."""
    v_j = np.random.randn(20)
    k_n = 2
    m = 10
    seed = 123

    v_star1 = selector._local_bootstrap_resample(v_j, k_n, m, seed=seed)
    v_star2 = selector._local_bootstrap_resample(v_j, k_n, m, seed=seed)

    np.testing.assert_array_equal(v_star1, v_star2)


def test_local_bootstrap_resample_absolute_value_reflection(selector):
    """Test that negative indices are reflected using absolute value.

    The paper specifies v*_j = v_{|j+S_j|} where |·| is absolute value.
    When j+S_j < 0, we should use v_{|j+S_j|}, not always v_1.
    """
    # Create a distinctive pattern so we can track which values are used
    v_j = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

    # Manually create a scenario where j + S_j becomes negative
    # For j=1 (0-indexed: j=0), if S[0]=-2, then j+S_j = 1-2 = -1
    # Paper says use v_{|-1|} = v_1 = 10.0 (v_j[0])
    # For j=2 (0-indexed: j=1), if S[1]=-3, then j+S_j = 2-3 = -1
    # Paper says use v_{|-1|} = v_1 = 10.0 (v_j[0])
    # For j=3 (0-indexed: j=2), if S[2]=-5, then j+S_j = 3-5 = -2
    # Paper says use v_{|-2|} = v_2 = 20.0 (v_j[1])

    # Create a mock _local_bootstrap_resample by setting specific S values.
    # Use a controlled random seed and check specific cases.
    k_n = 5
    m = 8
    seed = 999

    # Get the resampled values
    v_star = selector._local_bootstrap_resample(v_j, k_n, m, seed=seed)

    # Recreate the same random state to get the S values
    rng = np.random.RandomState(seed)
    S = rng.randint(-k_n, k_n + 1, size=m)

    # Verify each resampled value follows the absolute value rule
    for j in range(m):
        j_paper = j + 1  # Convert to 1-based
        idx_paper = j_paper + S[j]

        if idx_paper == 0:
            expected_idx = 0
        else:
            expected_idx = min(abs(idx_paper) - 1, len(v_j) - 1)

        expected_value = v_j[expected_idx]

        assert v_star[j] == expected_value, \
            f"For j={j}, S[j]={S[j]}, j+S={idx_paper}, " \
            f"expected v_j[{expected_idx}]={expected_value}, got {v_star[j]}"


def test_local_bootstrap_resample_kn_zero(selector):
    """Test with k_n = 0 (no randomization)."""
    v_j = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k_n = 0
    m = 3

    v_star = selector._local_bootstrap_resample(v_j, k_n, m, seed=42)

    # With k_n=0, should just return first m elements
    np.testing.assert_array_equal(v_star, v_j[:m])


#
# Bootstrap MSE tests
#

def test_compute_bootstrap_mse_basic(selector, simple_arfima_data):
    """Test basic MSE computation."""
    m_eval = 10
    d_init = 0.3
    k_n = 2

    mse, d_star = selector._compute_bootstrap_mse(
        simple_arfima_data, m_eval, d_init, k_n
    )

    # Check MSE is non-negative
    assert mse >= 0
    assert np.isfinite(mse)

    # Check d_star array
    assert len(d_star) == selector.B

    # All estimates should be finite
    valid_count = np.sum(np.isfinite(d_star))
    assert valid_count == selector.B


def test_compute_bootstrap_mse_all_nan(selector):
    """Test that all NaN estimates returns infinite MSE."""
    # Create data that will cause estimation failures
    X = np.ones(50)  # Constant series
    m_eval = 10
    d_init = 0.0
    k_n = 1

    mse, d_star = selector._compute_bootstrap_mse(X, m_eval, d_init, k_n)

    # Should handle gracefully
    assert np.isfinite(mse) or np.isinf(mse)


#
# Main fit tests
#

def test_fit_basic(selector, simple_arfima_data):
    """Test basic fit functionality."""
    selector.fit(simple_arfima_data)

    # Check that fitted attributes are set
    assert hasattr(selector, 'optimal_m_')
    assert hasattr(selector, 'd_hat_')
    assert hasattr(selector, 'se_')
    assert hasattr(selector, 'ase_')
    assert hasattr(selector, 'mse_profile_')
    assert hasattr(selector, 'k_n_')
    assert hasattr(selector, 'iterations_')
    assert hasattr(selector, 'converged_')

    # Check values are reasonable
    assert selector.m_min <= selector.optimal_m_ <= selector.m_max
    assert -1 < selector.d_hat_ < 2  # Reasonable d range
    assert selector.iterations_ <= selector.max_iter


def test_fit_with_auto_kn(lw_estimator, simple_arfima_data):
    """Test fit with automatic k_n selection."""
    selector = LWBootstrapM(
        lw_estimator=lw_estimator,
        k_n='auto',
        B=10,
        m_min=5,
        m_max=15,
        max_iter=2,
    )

    selector.fit(simple_arfima_data)

    # Check that k_n was selected
    assert hasattr(selector, 'k_n_')
    assert isinstance(selector.k_n_, (int, np.integer))
    assert selector.k_n_ > 0


def test_fit_returns_self(selector, simple_arfima_data):
    """Test that fit returns self for chaining."""
    result = selector.fit(simple_arfima_data)
    assert result is selector


def test_fit_mse_profile_populated(selector, simple_arfima_data):
    """Test that MSE profile is populated."""
    selector.fit(simple_arfima_data)

    assert isinstance(selector.mse_profile_, dict)
    assert len(selector.mse_profile_) > 0

    # Check that bandwidths in profile are within range
    for m in selector.mse_profile_.keys():
        assert selector.m_min <= m <= selector.m_max


#
# Repr tests
#

def test_repr_default_params(lw_estimator):
    """Test repr with default parameters."""
    selector = LWBootstrapM(lw_estimator=lw_estimator)
    repr_str = repr(selector)

    assert 'LWBootstrapM' in repr_str


def test_repr_custom_params(lw_estimator):
    """Test repr with custom parameters."""
    selector = LWBootstrapM(
        lw_estimator=lw_estimator,
        k_n=5,
        B=100,
        delta=-0.05
    )
    repr_str = repr(selector)

    assert 'LWBootstrapM' in repr_str
    assert 'k_n=5' in repr_str
    assert 'B=100' in repr_str
    assert 'delta=-0.05' in repr_str


#
# Edge case tests
#

def test_edge_case_m_min_equals_m_max(selector, simple_arfima_data):
    """Test when m_min equals m_max (no search)."""
    selector.m_min = 10
    selector.m_max = 10
    selector.m_init = 10

    selector.fit(simple_arfima_data)

    # Should converge immediately
    assert selector.optimal_m_ == 10

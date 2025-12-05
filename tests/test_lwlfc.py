import json
import os

import numpy as np
import pytest

from pyelw import LW, LWLFC
from pyelw.simulate import arfima


@pytest.fixture
def estimator():
    """Create LWLFC estimator instance."""
    return LWLFC()


@pytest.fixture
def estimator_noise():
    """Create LWPLFC estimator instance (with noise correction)."""
    return LWLFC(noise=True)


# =============================================================================
# Basic functionality tests
# =============================================================================

def test_init_default():
    """Test default initialization."""
    est = LWLFC()
    assert est.bounds == (-1.0, 2.2)
    assert est.noise is False


def test_init_custom():
    """Test custom initialization."""
    est = LWLFC(bounds=(-1.0, 1.0), noise=True)
    assert est.bounds == (-1.0, 1.0)
    assert est.noise is True


def test_repr():
    """Test string representation."""
    est = LWLFC()
    assert repr(est) == "LWLFC()"

    est = LWLFC(bounds=(-1.0, 1.0))
    assert "bounds=(-1.0, 1.0)" in repr(est)

    est = LWLFC(noise=True)
    assert "noise=True" in repr(est)


# =============================================================================
# Fit method tests
# =============================================================================

def test_fit_returns_self(estimator):
    """Test that fit() returns self."""
    np.random.seed(42)
    x = np.random.randn(256)
    result = estimator.fit(x)
    assert result is estimator


def test_fit_attributes(estimator):
    """Test that fit() sets all required attributes."""
    np.random.seed(42)
    x = np.random.randn(256)
    estimator.fit(x)

    assert hasattr(estimator, 'd_hat_')
    assert hasattr(estimator, 'theta_')
    assert hasattr(estimator, 'se_')
    assert hasattr(estimator, 'ase_')
    assert hasattr(estimator, 'n_')
    assert hasattr(estimator, 'm_')
    assert hasattr(estimator, 'objective_')
    assert hasattr(estimator, 'nfev_')
    assert hasattr(estimator, 'method_')


def test_fit_noise_attributes(estimator_noise):
    """Test that LWPLFC fit() sets noise-specific attributes."""
    np.random.seed(42)
    x = np.random.randn(256)
    estimator_noise.fit(x)

    assert hasattr(estimator_noise, 'theta_noise_')
    assert estimator_noise.method_ == 'lwplfc'


def test_fit_default_bandwidth(estimator):
    """Test default bandwidth is n^0.8."""
    np.random.seed(42)
    n = 256
    x = np.random.randn(n)
    estimator.fit(x)

    expected_m = int(n**0.8)
    assert estimator.m_ == expected_m


def test_fit_custom_bandwidth(estimator):
    """Test custom bandwidth."""
    np.random.seed(42)
    x = np.random.randn(256)
    m = 50
    estimator.fit(x, m=m)
    assert estimator.m_ == m


# =============================================================================
# Estimate method tests (backward compatibility API)
# =============================================================================

def test_estimate_returns_dict(estimator):
    """Test that estimate() returns a dictionary."""
    np.random.seed(42)
    x = np.random.randn(256)
    result = estimator.estimate(x)

    assert isinstance(result, dict)
    assert 'd_hat' in result
    assert 'theta' in result
    assert 'se' in result
    assert 'ase' in result
    assert 'n' in result
    assert 'm' in result
    assert 'objective' in result
    assert 'nfev' in result
    assert 'method' in result


# =============================================================================
# Tests with ARFIMA processes (no contamination)
# =============================================================================

@pytest.mark.parametrize("d_true", [0.0, 0.1, 0.2, 0.3, 0.4])
def test_arfima_estimation(d_true):
    """Test LWLFC estimation with pure ARFIMA(0,d,0) processes."""
    np.random.seed(42)
    n = 1024
    x = arfima(n, d_true, seed=42)

    est = LWLFC()
    est.fit(x, m=int(n**0.7))

    # Should recover d reasonably well with clean data
    error = abs(est.d_hat_ - d_true)
    assert error < 0.05, f"d_hat={est.d_hat_:.4f}, d_true={d_true}, error={error:.4f}"

    # SE should be positive and finite
    assert est.se_ > 0
    assert np.isfinite(est.se_)
    assert est.ase_ > 0
    assert np.isfinite(est.ase_)


# =============================================================================
# Tests with random level shift contamination
# =============================================================================

def generate_rls(n, p, sigma_eta=1.0, seed=None):
    """
    Helper for generating a random level shift (RLS) process.

    Following Definition 1 in Hou and Perron (2014, p. 311):
    u_t = sum_{s=1}^t delta_{T,s} where delta_{T,t} = pi_{T,t} * eta_t
    with eta_t ~ iid N(0, sigma_eta^2) and pi_{T,t} ~ Bernoulli(p/T, 1).

    Parameters
    ----------
    n : int
        Sample size.
    p : float
        Expected number of level shifts over the entire sample
        (probability = p/n per period).
    sigma_eta : float
        Standard deviation of level shift magnitudes.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        RLS process of length n.
    """
    if seed is not None:
        np.random.seed(seed)

    # Bernoulli draws for level shift occurrences
    probs = np.random.random(n)
    shifts = probs < (p / n)

    # Level shift magnitudes
    eta = np.random.normal(0, sigma_eta, n)

    # Cumulative sum of shifts
    delta = shifts * eta
    u = np.cumsum(delta)

    return u


def test_rls_helper():
    """Test RLS generation helper function."""
    n = 1000
    p = 10
    u = generate_rls(n, p, sigma_eta=1.0, seed=42)

    assert len(u) == n
    assert np.isfinite(u).all()


@pytest.mark.parametrize("p", [0, 5, 10, 20])
def test_short_memory_with_rls(p):
    """
    Test LWLFC with short-memory process + RLS.

    This is the main use case from Table 1 of Hou and Perron (2014).
    With RLS contamination, standard LW would give spurious long memory,
    but LWLFC should estimate d close to 0.
    """
    np.random.seed(42)
    n = 512
    d_true = 0.0

    # Short-memory process + RLS contamination
    y = arfima(n, d_true, seed=42)
    u = generate_rls(n, p, sigma_eta=1.0, seed=43)
    z = y + u

    m = int(n**0.8)
    lw_est = LW().fit(z, m=m)
    lwlfc_est = LWLFC().fit(z, m=m)

    # Sanity checks
    assert np.isfinite(lw_est.d_hat_)
    assert np.isfinite(lwlfc_est.d_hat_)
    assert lwlfc_est.theta_ >= 0  # theta should be non-negative

    # LWLFC should generally give estimates closer to 0 than standard LW
    assert abs(lwlfc_est.d_hat_) < abs(lw_est.d_hat_)


@pytest.mark.parametrize("d_true", [0.0, 0.2, 0.4, 0.45, 0.6])
def test_long_memory_with_rls(d_true):
    """
    Test LWLFC with long-memory process + RLS.
    Hou and Perron (2014, Tables 2-3).
    """
    np.random.seed(42)
    n = 1024
    p = 10

    # Long-memory process + RLS contamination
    y = arfima(n, d_true, seed=42)
    u = generate_rls(n, p, sigma_eta=1.0, seed=43)
    z = y + u

    # Fit model
    est = LWLFC().fit(z, m=int(n**0.8))

    # Should recover d reasonably
    assert np.isfinite(est.d_hat_)
    error = abs(est.d_hat_ - d_true)
    # Generous tolerance for single realization with contamination
    assert error < 0.15, f"d_hat={est.d_hat_:.4f}, d_true={d_true}, error={error:.4f}"


# =============================================================================
# Tests for LWPLFC (with additive noise)
# =============================================================================

def test_lwplfc_with_noise():
    """Test LWPLFC with additive noise contamination."""
    np.random.seed(42)
    n = 4096
    d_true = 0.3

    # Long-memory + additive noise
    y = arfima(n, d_true, seed=42)
    w = np.random.normal(0, 2.0, n)  # Strong noise
    z = y + w

    # Fit with defaults
    est = LWLFC(noise=True).fit(z)

    # Basic checks
    assert est.method_ == 'lwplfc'
    assert np.isfinite(est.d_hat_)
    assert est.theta_ >= 0
    assert est.theta_noise_ >= 0

    # Should recover d reasonably
    assert np.isfinite(est.d_hat_)
    error = abs(est.d_hat_ - d_true)
    # Tight tolerance for large sample size
    assert error < 0.01, f"d_hat={est.d_hat_:.4f}, d_true={d_true}, error={error:.4f}"


def test_lwplfc_with_rls_and_noise():
    """Test LWPLFC with both RLS and additive noise."""
    np.random.seed(42)
    n = 4096
    d_true = 0.3
    p = 10

    # Long-memory + RLS + noise
    y = arfima(n, d_true, seed=42)
    u = generate_rls(n, p, sigma_eta=1.0, seed=43)
    w = np.random.normal(0, 1.0, n)
    z = y + u + w

    # Fit model
    est = LWLFC(noise=True).fit(z, m=int(n**0.8))

    # Basic checks
    assert est.method_ == 'lwplfc'
    assert np.isfinite(est.d_hat_)
    assert est.theta_ >= 0
    assert est.theta_noise_ >= 0

    # Should recover d reasonably
    assert np.isfinite(est.d_hat_)
    error = abs(est.d_hat_ - d_true)
    # Tight tolerance for large sample size
    assert error < 0.015, f"d_hat={est.d_hat_:.4f}, d_true={d_true}, error={error:.4f}"


# =============================================================================
# Edge case tests
# =============================================================================

def test_short_series():
    """Test with short time series."""
    np.random.seed(42)
    x = np.random.randn(10)
    est = LWLFC().fit(x)
    assert np.isfinite(est.d_hat_)


def test_constant_series():
    """Test with constant series (should handle gracefully)."""
    x = np.ones(256)
    est = LWLFC().fit(x)
    assert hasattr(est, 'd_hat_')  # Should not crash


def test_trending_series():
    """Test with deterministic trend."""
    np.random.seed(42)
    n = 512
    t = np.arange(n)
    trend = 0.01 * t
    noise = np.random.randn(n)
    x = trend + noise

    est = LWLFC().fit(x)
    assert np.isfinite(est.d_hat_)
    assert est.theta_ >= 0


# =============================================================================
# Verbose output tests
# =============================================================================

def test_verbose_output(capsys):
    """Test verbose mode produces output."""
    np.random.seed(42)
    x = np.random.randn(256)

    est = LWLFC()
    est.fit(x, verbose=True)

    captured = capsys.readouterr()
    assert "LWLFC Estimation Results" in captured.out
    assert "d_hat" in captured.out
    assert "theta" in captured.out


# =============================================================================
# Objective function tests
# =============================================================================

def test_objective_finite(estimator):
    """Test that objective returns finite values for valid inputs."""
    np.random.seed(42)
    x = np.random.randn(256)
    m = int(256**0.8)
    data = estimator.prepare_data(x, m)
    params = np.array([0.2, 0.1])
    obj = estimator.objective(params, data)
    assert np.isfinite(obj)


def test_objective_large_theta(estimator):
    """Test objective with large theta."""
    np.random.seed(42)
    x = np.random.randn(256)
    m = int(256**0.8)
    data = estimator.prepare_data(x, m)
    params = np.array([0.2, 1e10])
    obj = estimator.objective(params, data)
    assert isinstance(obj, (float, np.floating))


# =============================================================================
# Test against R LongMemoryTS Hou.Perron baseline results
# =============================================================================

@pytest.fixture
def nile_data():
    """Load Nile river flow dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "nile.csv")
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = [float(row['nile']) for row in reader]
    return np.array(data)


@pytest.fixture
def sealevel_data():
    """Load sea level dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "sealevel.csv")
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = [float(row['Sea']) for row in reader]
    return np.array(data)


def _load_r_lwlfc_cases():
    """Load R Hou.Perron baseline results and convert to test cases."""
    json_path = os.path.join(os.path.dirname(__file__), "r_lwlfc.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r') as f:
        r_results = json.load(f)
    test_cases = []
    for dataset, dataset_data in r_results.items():
        for size, case_data in dataset_data.items():
            if 'error' in case_data:
                continue
            test_case = {
                'name': f"{dataset}:{size}",
                'dataset': dataset,
                'size': size,
                'n': case_data['n'],
                'm': case_data['m'],
                'expected_d_hat': case_data['d_hat'],
                'expected_theta': case_data['theta'],
                'expected_obj': case_data['obj'],
                'expected_ase': case_data['ase'],
            }
            test_cases.append(test_case)
    return test_cases


R_LWLFC_CASES = _load_r_lwlfc_cases()


@pytest.mark.parametrize("case", R_LWLFC_CASES)
def test_r_lwlfc_baseline(case, nile_data, sealevel_data):
    """Test LWLFC estimator against R Hou.Perron results."""

    # Extract test case parameters
    dataset = case['dataset']
    expected_d_hat = case['expected_d_hat']
    expected_theta = case['expected_theta']
    expected_obj = case['expected_obj']
    expected_ase = case['expected_ase']
    n = case['n']
    m = case['m']

    # Get dataset from fixtures
    if dataset == 'nile':
        series = nile_data
    elif dataset == 'sealevel':
        series = sealevel_data
    else:
        pytest.skip(f"Unknown dataset: {dataset}")

    assert len(series) == n, f"Dataset length mismatch for {dataset}: {len(series)} vs {n}"

    # Run LWLFC estimation with bounds matching R implementation
    # R uses lower=c(-0.4999, 0) and upper=c(0.99, 10000)
    lwlfc = LWLFC(bounds=(-0.4999, 0.99))
    result = lwlfc.estimate(series, m=m, verbose=False)

    # Check basic properties
    assert result['n'] == n, f"Sample size mismatch for {case['name']}: {result['n']} vs {n}"
    assert result['m'] == m, f"Bandwidth mismatch for {case['name']}: {result['m']} vs {m}"

    # Check that results are finite
    assert np.isfinite(result['d_hat']), f"Non-finite d_hat for {case['name']}"
    assert np.isfinite(result['se']), f"Non-finite se for {case['name']}"
    assert np.isfinite(result['objective']), f"Non-finite objective for {case['name']}"

    # Compute absolute or relative errors
    d_error = abs(result['d_hat'] - expected_d_hat)
    theta_rel_error = abs(result['theta'] - expected_theta) / abs(expected_theta)
    obj_error = abs(result['objective'] - expected_obj)
    ase_error = abs(result['ase'] - expected_ase)

    # Print comparison for debugging (pytest with -s flag)
    print(f"\n{dataset} (m={m}):")
    print(f"  d_hat: Python={result['d_hat']:10.6f}, R={expected_d_hat:10.6f}, diff={d_error:.2e}")
    print(f"  theta: Python={result['theta']:10.4f}, R={expected_theta:10.4f}, diff={theta_rel_error:.2e}")
    print(f"  obj:   Python={result['objective']:10.6f}, R={expected_obj:10.6f}, diff={obj_error:.2e}")
    print(f"  ase:   Python={result['ase']:10.6f}, R={expected_ase:10.6f}, diff={ase_error:.2e}")

    # Check d estimate
    atol_d = 0.01
    assert d_error <= atol_d, (
        f"d_hat mismatch for {case['name']}: "
        f"Python={result['d_hat']:.6f}, R={expected_d_hat:.6f}, error={d_error:.6f}"
    )

    # Check theta estimate
    # We use a loose relative tolerance since theta is a poorly identified nuisance parameter.
    rtol_theta = 0.1
    assert theta_rel_error <= rtol_theta, (
        f"theta mismatch for {case['name']}: "
        f"Python={result['theta']:.6f}, R={expected_theta:.6f}, error={theta_rel_error:.6f}"
    )

    # Check objective function value
    atol_obj = 1e-4
    assert obj_error <= atol_obj, (
        f"objective mismatch for {case['name']}: "
        f"Python={result['objective']:.6f}, R={expected_obj:.6f}, error={obj_error:.6f}"
    )

    # Check asymptotic standard error (deterministic formula, should match exactly)
    atol_ase = 1e-10
    assert ase_error <= atol_ase, (
        f"ase mismatch for {case['name']}: "
        f"Python={result['ase']:.6f}, R={expected_ase:.6f}, error={ase_error:.6f}"
    )

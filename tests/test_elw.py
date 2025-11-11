import os
import json
import pytest
import numpy as np

from pyelw import ELW
from pyelw.simulate import arfima


@pytest.fixture
def estimator():
    """Create ELW estimator instance."""
    return ELW()


@pytest.mark.parametrize(
    "spec", [
        (0.2,  3,  0.153705, 0.47560, 0.28870),
        (0.3,  7,  0.706223, 0.25540, 0.18900),
        (0.4,  13, 0.523264, 0.17170, 0.13870),
        (0.5,  25, 0.453753, 0.11060, 0.10000),
        (0.6,  49, 0.450481, 0.07844, 0.07143),
        (0.7,  94, 0.392375, 0.05216, 0.05157),
        (0.8, 180, 0.408864, 0.04007, 0.03727),
        (0.9, 346, 0.542664, 0.04032, 0.02688),
    ])
def test_nile(estimator, spec, nile_data):
    """Test against Stata whittle.ado estimates for nile dataset."""
    alpha, m, d_hat, se, ase = spec
    # Stata's whittle.ado aleays demeans the data
    result = estimator.estimate(nile_data, m=m, mean_est='mean')

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-5, atol=1e-8,
        err_msg=f"ELW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize(
    "spec", [
        (0.65,  68, 0.407459, 0.06243, 0.06063),
    ])
def test_baum_et_al_nile(estimator, spec, nile_data):
    """Test against Baum, Hurn, and Lindsay (2020), page 578."""
    alpha, m, d_hat, se, ase = spec
    # Stata's whittle.ado always demeans the data
    result = estimator.estimate(nile_data, m=m, mean_est='mean')

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-6, atol=1e-6,
        err_msg=f"ELW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize(
    "spec", [
        (0.60,  82, 0.888921, 0.04535, 0.05522),
        (0.65, 118, 0.801612, 0.03494, 0.04603),
        (0.70, 171, 0.773669, 0.02987, 0.03824),
        (0.75, 247, 0.784673, 0.02743, 0.03181),
        (0.80, 358, 0.881789, 0.03067, 0.02643),
    ])
def test_sealevel(estimator, spec, sealevel_data):
    """Test against Stata whittle.ado estimates for sealevel dataset."""
    alpha, m, d_hat, se, ase = spec
    # Stata's whittle.ado always demeans the data
    result = estimator.estimate(sealevel_data, m=m, mean_est='mean')

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-6, atol=1e-8,
        err_msg=f"ELW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize(
    "spec", [
        (0.65, 118, 0.801612, 0.03494, 0.04603),
    ])
def test_baum_et_al_sealevel(estimator, spec, sealevel_data):
    """Test against Baum, Hurn, and Lindsay (2020), page 578."""
    alpha, m, d_hat, se, ase = spec
    # Stata's whittle.ado always demeans the data
    result = estimator.estimate(sealevel_data, m=m, mean_est='mean')

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-6, atol=1e-8,
        err_msg=f"ELW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"ELW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize("d_true", [-2.5, -2.2, -1.2, -0.8, -0.3, 0.0, 0.2, 0.4, 1.2, 2.2, 2.8, 3.2, 3.5, 4.0])
def test_arfima(estimator, d_true):
    """Test estimation with ARFIMA(0,d,0) processes."""
    n = 5000
    seed = 42
    x = arfima(n, d_true, seed=seed)
    result = estimator.estimate(x, m=int(n**0.7), bounds=(-3, 4))

    # Estimates should be reasonably close to true value
    assert np.isfinite(result['d_hat'])
    error = abs(result['d_hat'] - d_true)
    print(f"Testing ARFIMA with d_true={d_true}, n={n}, seed={seed}")
    print(f"Estimated d: {result['d_hat']}, True d: {d_true}, Error: {error}")
    print(error)
    assert error < 0.05, f"ARFIMA method: d_hat={result['d_hat']}, d_true={d_true}"

    # Fisher SE and asymptotic SE should be similar for large n
    fisher_se = result['se']
    asymptotic_se = result['ase']
    assert fisher_se > 0
    assert asymptotic_se > 0
    assert np.isfinite(fisher_se)
    assert np.isfinite(asymptotic_se)
    diff = abs(fisher_se - asymptotic_se)
    assert diff < 0.05, f"Fisher SE and Asymptotic SE differ too much: {diff}"


def test_white_noise(estimator):
    """Test estimation on white noise (d=0)."""
    n = 1000
    seed = 42
    x = arfima(n, 0.0, sigma=1.0, seed=seed)  # White noise
    result = estimator.estimate(x)

    assert np.isfinite(result['d_hat'])
    assert abs(result['d_hat']) < 0.02  # Should be close to zero
    assert result['se'] > 0
    assert result['ase'] > 0


def test_differenced_random_walk(estimator):
    """Test estimation on differenced random walk."""
    n = 1000
    seed = 42

    # Generate I(1) process and difference it
    x_rw = arfima(n + 1, 1.0, sigma=1.0, seed=seed)
    x = np.diff(x_rw)  # Difference to make stationary
    result = estimator.estimate(x)

    # Differenced I(1) becomes I(0), so d should be near 0
    assert np.isfinite(result['d_hat'])
    assert abs(result['d_hat']) < 0.05  # Should be close to zero
    assert result['se'] > 0
    assert result['ase'] > 0


def test_objective_minimized(estimator):
    """Test that objective is minimized at estimated parameter."""
    n = 1000
    d_true = 0.25
    seed = 42
    x = arfima(n, d_true, seed=seed)
    m = int(n**0.7)
    result = estimator.estimate(x, m=m)
    d_hat = result['d_hat']

    # Test that objective is higher at nearby points
    delta = 0.05
    nearby_points = [d_hat - delta, d_hat + delta]
    obj_at_estimate = estimator.objective(d_hat, x, m)
    for d_test in nearby_points:
        obj_at_test = estimator.objective(d_test, x, m)
        assert obj_at_test >= obj_at_estimate, \
            f"Objective not minimized: f({d_hat})={obj_at_estimate}, f({d_test})={obj_at_test}"


def test_different_m_values(estimator):
    """Test stability across different m values."""
    n = 10000
    d_true = 0.3
    seed = 42
    x = arfima(n, d_true, seed=seed)

    # Test range of m values from n^0.5 to n^0.8
    m_values = [int(n**exp) for exp in np.linspace(0.5, 0.8, 5)]
    estimates = []
    for m in m_values:
        result = estimator.estimate(x, m=m)
        estimates.append(result['d_hat'])

    # Estimates should be reasonably similar
    estimate_range = max(estimates) - min(estimates)
    assert estimate_range < 0.1, f"Estimates vary too much across m values: {estimates}"


def test_fisher_vs_asymptotic_se(estimator):
    """Test similarity of Fisher Information and asymptotic SE."""
    n = 10000
    d_true = 0.3
    seed = 42

    x = arfima(n, d_true, seed=seed)
    result = estimator.estimate(x, m=int(n**0.7))
    fisher_se = result['se']
    asymptotic_se = result['ase']

    assert np.isfinite(fisher_se)
    assert np.isfinite(asymptotic_se)
    assert fisher_se > 0
    assert asymptotic_se > 0

    # They should be similar for large n
    diff = abs(fisher_se - asymptotic_se)
    assert diff < 0.1, \
        f"Fisher SE ({fisher_se}) and Asymptotic SE ({asymptotic_se}) differ too much: {diff}"


def test_se_scaling(estimator):
    """Standard errors should scale approximately as 1/sqrt(m)."""
    n = 20000
    d_true = 0.2
    seed = 42

    x = arfima(n, d_true, seed=seed)

    m_values = [int(n**0.6), int(n**0.7), int(n**0.8)]
    se_values = []
    ase_values = []
    for m in m_values:
        result = estimator.estimate(x, m=m)
        se_values.append((m, result['se']))
        ase_values.append((m, result['ase']))

    for i in range(len(ase_values) - 1):
        # Test asymptotic SE scaling (should be exact)
        m1, ase1 = ase_values[i]
        m2, ase2 = ase_values[i + 1]
        expected_ratio = np.sqrt(m2) / np.sqrt(m1)
        actual_ratio = ase1 / ase2
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-10,
                                   err_msg=f"Asymptotic SE scaling failed: {ase1} vs {ase2}")

        # Test Fisher SE scaling (should be approximately 1/sqrt(m))
        m1, se1 = se_values[i]
        m2, se2 = se_values[i+1]
        expected_ratio = np.sqrt(m2) / np.sqrt(m1)
        actual_ratio = se1 / se2
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-1,
                                   err_msg=f"Fisher SE scaling failed: {se1} vs {se2}")


#
# Test against R LongMemoryTS baseline results
#

# Load R test cases
def _load_r_elw_cases():
    """Load R LongMemoryTS baseline results and convert to test cases."""
    json_path = os.path.join(os.path.dirname(__file__), "r_elw.json")
    with open(json_path, 'r') as f:
        r_results = json.load(f)
    test_cases = []
    for dataset, dataset_data in r_results.items():
        for size, case_data in dataset_data.items():
            test_case = {
                'name': f"{dataset}:{size}",
                'dataset': dataset,
                'size': size,
                'n': case_data['n'],
                'm': case_data['m'],
                'expected_d_hat': case_data['d_hat'],
                'expected_se': case_data['se'],
            }
            test_cases.append(test_case)
    return test_cases


# Load test cases
R_ELW_CASES = _load_r_elw_cases()


@pytest.mark.parametrize("case", R_ELW_CASES)
def test_r_elw_baseline(case, nile_data, sealevel_data):
    """Test ELW estimator against R LongMemoryTS ELW results."""

    # Extract test case parameters
    dataset = case['dataset']
    expected_d_hat = case['expected_d_hat']
    expected_se = case['expected_se']
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

    # Run exact local Whittle estimation (no initial grid search to match R)
    elw = ELW(n_grid=0)
    result = elw.estimate(series, m=m, verbose=False)

    # Check basic properties
    assert result['n'] == n, f"Sample size mismatch for {case['name']}: {result['n']} vs {n}"
    assert result['m'] == m, f"Bandwidth mismatch for {case['name']}: {result['m']} vs {m}"

    # Check that results are finite
    assert np.isfinite(result['d_hat']), f"Non-finite d_hat for {case['name']}"
    assert np.isfinite(result['se']), f"Non-finite se for {case['name']}"

    # Compute differences
    d_error = abs(result['d_hat'] - expected_d_hat)
    se_error = abs(result['ase'] - expected_se)

    # Print comparison for debugging (pytest with -s flag)
    print(f"\n{dataset} (m={m}):")
    print(f"  d_hat: Python={result['d_hat']:10.6f}, R={expected_d_hat:10.6f}, diff={d_error:.2e}")
    print(f"  se:    Python={result['se']:10.6f}, R={expected_se:10.6f}, diff={se_error:.2e}")

    # Check d_hat estimate
    atol_d = 1e-4
    assert d_error <= atol_d, \
        f"d_hat mismatch for {case['name']}: Python={result['d_hat']:.6f}, R={expected_d_hat:.6f}, error={d_error:.6f}"

    # Check asymptotic standard errors
    atol_se = 1e-8
    assert se_error <= atol_se, \
        f"se mismatch for {case['name']}: Python={result['ase']:.6f}, R={expected_se:.6f}, error={se_error:.6f}"


@pytest.mark.parametrize("case", R_ELW_CASES)
def test_r_elw_baseline_robust(case, nile_data, sealevel_data):
    """Test that robust ELW optimization finds better or equal solutions than
    R LongMemoryTS.

    The LongMemoryTS ELW implementation can get stuck in local minima. This
    test verifies that our robust optimization (n_grid > 0) finds solutions
    with objective function values at least as good as, and often better than,
    the non-robust method that matches R's results.

    For the Nile dataset, robust optimization detects and corrects local minima:
    - Case 0 (m=49): Improvement of 1.02 in objective, d changes from 0.958585 to 0.016639
    - Case 1 (m=94): Improvement of 0.64 in objective, d changes from 0.808957 to 0.017334
    - Case 2 (m=180): Improvement of 0.21 in objective, d changes from 0.773317 to 0.019372
    """
    # Extract test case parameters
    dataset = case['dataset']
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

    # Run non-robust optimization (matching R behavior)
    elw_standard = ELW(n_grid=0)
    elw_standard.fit(series, m=m)
    d_standard = elw_standard.d_hat_
    obj_standard = elw_standard.objective_

    # Run robust optimization (default behavior)
    elw_robust = ELW(n_grid=20)
    elw_robust.fit(series, m=m)
    d_robust = elw_robust.d_hat_
    obj_robust = elw_robust.objective_

    # Verify both solutions are finite
    assert np.isfinite(d_standard), f"Non-finite d_hat from standard optimization for {case['name']}"
    assert np.isfinite(d_robust), f"Non-finite d_hat from robust optimization for {case['name']}"
    assert np.isfinite(obj_standard), f"Non-finite objective from standard optimization for {case['name']}"
    assert np.isfinite(obj_robust), f"Non-finite objective from robust optimization for {case['name']}"

    # Calculate improvement
    obj_improvement = obj_standard - obj_robust
    d_difference = abs(d_standard - d_robust)

    # Print comparison for debugging
    print(f"\n{dataset} (m={m}) - Robust Optimization Test:")
    print(f"  Standard (R-like):  d={d_standard:10.6f},  obj={obj_standard:10.6f}")
    print(f"  Robust:             d={d_robust:10.6f},  obj={obj_robust:10.6f}")

    # Robust optimization should find a solution at least as good as standard
    # (lower or equal objective function value)
    assert obj_robust <= obj_standard + 1e-6, \
        f"Robust optimization found worse solution for {case['name']}: " \
        f"obj_robust={obj_robust:.6f} > obj_standard={obj_standard:.6f}"


def test_constructor_defaults():
    """Test constructor with default parameters."""
    elw = ELW()
    assert elw.bounds == (-1.0, 2.2)
    assert elw.mean_est == 'none'
    assert elw.n_grid == 20
    assert elw._default_bounds == (-1.0, 2.2)
    assert elw._default_mean_est == 'none'


def test_constructor_custom_params():
    """Test constructor with custom parameters."""
    elw = ELW(bounds=(-0.5, 1.5), mean_est='mean')
    assert elw.bounds == (-0.5, 1.5)
    assert elw.mean_est == 'mean'


def test_repr_default_params():
    """Test __repr__ with default parameters."""
    elw = ELW()
    assert repr(elw) == 'ELW()'
    assert str(elw) == 'ELW()'


def test_repr_custom_params():
    """Test __repr__ with custom parameters."""
    elw = ELW(bounds=(-0.5, 1.5))
    assert repr(elw) == "ELW(bounds=(-0.5, 1.5))"

    elw = ELW(mean_est="mean")
    assert repr(elw) == "ELW(mean_est='mean')"

    elw = ELW(bounds=(-0.5, 1.5), mean_est="init")
    assert repr(elw) == "ELW(bounds=(-0.5, 1.5), mean_est='init')"


def test_fit_basic():
    """Test basic fit functionality."""
    np.random.seed(42)
    X = np.random.randn(100)
    elw = ELW()
    result = elw.fit(X, m=20)

    # Should return self
    assert result is elw

    # Should have fitted attributes
    assert hasattr(elw, 'd_hat_')
    assert hasattr(elw, 'se_')
    assert hasattr(elw, 'ase_')
    assert hasattr(elw, 'n_')
    assert hasattr(elw, 'm_')
    assert hasattr(elw, 'objective_')
    assert hasattr(elw, 'nfev_')

    # Check values are reasonable
    assert elw.n_ == len(X)
    assert elw.m_ == 20
    assert np.isfinite(elw.d_hat_)


def test_fit_method_chaining():
    """Test method chaining with fit."""
    np.random.seed(42)
    X = np.random.randn(100)
    d_hat = ELW(bounds=(-0.5, 1.5)).fit(X, m=20).d_hat_
    assert np.isfinite(d_hat)


def test_backward_compatibility_estimate():
    """Test that estimate() API works."""
    np.random.seed(42)
    X = np.random.randn(100)
    elw = ELW()
    result = elw.estimate(X, m=20, bounds=(-0.5, 1.5), mean_est='mean')

    # Should return dict
    assert isinstance(result, dict)
    assert 'd_hat' in result
    assert 'se' in result
    assert 'method' in result
    assert result['method'] == 'elw'


def test_estimate_parameter_override():
    """Test that estimate() parameters temporarily override constructor params."""
    np.random.seed(42)
    X = np.random.randn(100)
    elw = ELW(bounds=(-1.0, 2.2), mean_est='none')

    # Use different parameters in estimate()
    _ = elw.estimate(X, m=20, bounds=(-0.5, 1.5), mean_est='mean')

    # Constructor params should be restored
    assert elw.bounds == (-1.0, 2.2)
    assert elw.mean_est == 'none'


@pytest.mark.slow
def test_elw_auto_m_basic(arfima_data_auto):
    """Test that ELW.fit(m='auto') works and sets bootstrap attributes."""
    x, d_true = arfima_data_auto

    elw = ELW()
    elw.fit(x, m='auto')

    # Check that bootstrap attributes are set
    assert hasattr(elw, 'bootstrap_m_optimal_m_')
    assert hasattr(elw, 'bootstrap_m_iterations_')
    assert hasattr(elw, 'bootstrap_m_mse_profile_')
    assert hasattr(elw, 'bootstrap_m_k_n_')

    # Check that standard fitted attributes are set
    assert hasattr(elw, 'd_hat_')
    assert hasattr(elw, 'se_')
    assert hasattr(elw, 'm_')
    assert hasattr(elw, 'n_')

    # Check that optimal_m was used
    assert elw.m_ == elw.bootstrap_m_optimal_m_

    # Check that estimate is reasonable
    assert np.isfinite(elw.d_hat_)
    assert abs(elw.d_hat_ - d_true) < 0.3  # Loose bound

    # Use mean estimation
    elw = ELW(mean_est='mean')
    elw.fit(x, m='auto')

    # Check that bootstrap attributes are set
    assert hasattr(elw, 'bootstrap_m_optimal_m_')

    # Check that m matches optimal_m
    assert elw.m_ == elw.bootstrap_m_optimal_m_

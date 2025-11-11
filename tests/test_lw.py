import os
import json
import pytest
import numpy as np

from pyelw import LW
from pyelw.simulate import arfima


@pytest.fixture
def estimator():
    """Create LW estimator instance."""
    return LW()


@pytest.mark.parametrize(
    "spec", [
        (0.40,  13, 0.538433, 0.18060, 0.13870),
        (0.50,  25, 0.466848, 0.11390, 0.10000),
        (0.60,  49, 0.459277, 0.07914, 0.07143),
        (0.70,  94, 0.385763, 0.05091, 0.05157),
        (0.80, 180, 0.376356, 0.03674, 0.03727),
        (0.90, 346, 0.366238, 0.02649, 0.02688),
    ])
def test_lw_nile(estimator, spec, nile_data):
    """Test against Stata whittle.ado estimates for nile dataset."""
    alpha, m, d_hat, se, ase = spec
    result = estimator.estimate(nile_data, m=m)

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-5, atol=1e-9,
        err_msg=f"LW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check Fisher standard errors
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"LW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check asymptotic standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"LW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize(
    "spec", [
        (0.50,  25, 0.466848, 0.11390, 0.10000),
        (0.55,  35, 0.469123, 0.09495, 0.08452),
        (0.60,  49, 0.459277, 0.07914, 0.07143),
        (0.65,  68, 0.409044, 0.06212, 0.06063),
        (0.70,  94, 0.385763, 0.05091, 0.05157),
    ])
def test_lw_baum_et_al_nile(estimator, spec, nile_data):
    """Test against Baum, Hurn, and Lindsay (2020), page 577."""
    alpha, m, d_hat, se, ase = spec
    result = estimator.estimate(nile_data, m=m)

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-5, atol=1e-9,
        err_msg=f"LW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check Fisher standard errors
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"LW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check asymptotic standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"LW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize(
    "spec", [
        (0.60,  82, 0.932228, 0.05769, 0.05522),
        (0.65, 118, 0.859211, 0.04384, 0.04603),
        (0.70, 171, 0.854608, 0.03705, 0.03824),
        (0.75, 247, 0.854038, 0.03113, 0.03181),
        (0.80, 358, 0.898448, 0.02806, 0.02643),
    ])
def test_lw_sealevel(estimator, spec, sealevel_data):
    """Test against Stata whittle.ado estimates for sealevel dataset."""
    alpha, m, d_hat, se, ase = spec
    result = estimator.estimate(sealevel_data, m=m)

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-5, atol=1e-9,
        err_msg=f"LW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check Fisher standard errors
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"LW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check asymptotic standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"LW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize(
    "spec", [
        (0.65, 118, 0.859211, 0.04384, 0.04603),
    ])
def test_lw_baum_et_al_sealevel(estimator, spec, sealevel_data):
    """Test against Baum, Hurn, and Lindsay (2020), page 578."""
    alpha, m, d_hat, se, ase = spec
    result = estimator.estimate(sealevel_data, m=m)

    # Check estimate
    np.testing.assert_allclose(
        result['d_hat'], d_hat, rtol=1e-5, atol=1e-9,
        err_msg=f"LW d estimate differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check Fisher standard errors
    np.testing.assert_allclose(
        result['se'], se, rtol=1e-3, atol=1e-8,
        err_msg=f"LW Fisher SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )
    # Check asymptotic standard errors
    np.testing.assert_allclose(
        result['ase'], ase, rtol=1e-3, atol=1e-8,
        err_msg=f"LW asymptotic SE differs from Stata benchmark (alpha = {alpha}, m = {m})"
    )


@pytest.mark.parametrize("d_true", [-0.4, -0.3, 0.0, 0.2, 0.4])
def test_lw_arfima(estimator, d_true):
    """Test estimation with ARFIMA(0,d,0) processes."""
    n = 20000
    seed = 1000 + int(d_true * 100)  # Different seed for each d_true
    x = arfima(n, d_true, seed=seed)
    result = estimator.estimate(x, m=int(n**0.7))

    # Estimates should be reasonably close to true value
    assert np.isfinite(result['d_hat'])
    error = abs(result['d_hat'] - d_true)
    print(f"Testing ARFIMA with d_true={d_true}, n={n}, seed={seed}")
    print(f"Estimated d: {result['d_hat']}, True d: {d_true}, Error: {error}")
    print(error)
    assert error < 0.05, f"ARFIMA estimation: d_hat={result['d_hat']}, d_true={d_true}"

    # Fisher SE should be positive and finite
    fisher_se = result['se']
    assert fisher_se > 0
    assert np.isfinite(fisher_se)


@pytest.mark.parametrize("d_true", [-0.4, 0.0, 0.5, 0.9, 1.2, 1.4, 1.6, 2.0])
@pytest.mark.parametrize("taper", ['hc', 'cosine', 'kolmogorov', 'bartlett'])
def test_taper_arfima(estimator, d_true, taper):
    """Test tapered estimators with ARFIMA(0,d,0) processes."""
    n = 20000
    seed = 42 + int(d_true * 100)  # Different seed for each d_true
    x = arfima(n, d_true, seed=seed)
    result = estimator.estimate(x, m=int(n**0.7), taper=taper)

    # Estimates should be reasonably close to true value
    assert np.isfinite(result['d_hat'])
    error = abs(result['d_hat'] - d_true)
    print(f"Testing ARFIMA with d_true={d_true}, n={n}, seed={seed}")
    print(f"Estimated d: {result['d_hat']}, True d: {d_true}, Error: {error}")
    print(error)
    assert error < 0.05, f"ARFIMA estimation with taper='{taper}': d_hat={result['d_hat']}, d_true={d_true}"

    # Fisher SE should be positive and finite
    fisher_se = result['se']
    assert fisher_se > 0
    assert np.isfinite(fisher_se)


def test_lw_white_noise(estimator):
    """Test estimation on white noise (d=0)."""
    n = 1000
    seed = 42
    x = arfima(n, 0.0, sigma=1.0, seed=seed)  # White noise
    result = estimator.estimate(x)

    assert np.isfinite(result['d_hat'])
    assert abs(result['d_hat']) < 0.02  # Should be close to zero
    assert result['se'] > 0


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


def test_objective_minimized(estimator):
    """Test that objective is minimized at estimated parameter."""
    n = 1000
    d_true = 0.25
    seed = 42
    x = arfima(n, d_true, seed=seed)
    data = estimator.prepare_data(x, int(n**0.7))
    result = estimator.estimate(x, m=int(n**0.7))
    d_hat = result['d_hat']

    # Test that objective is higher at nearby points
    delta = 0.05
    nearby_points = [d_hat - delta, d_hat + delta]
    obj_at_estimate = estimator.objective(d_hat, data)
    for d_test in nearby_points:
        obj_at_test = estimator.objective(d_test, data)
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


def test_se_scaling(estimator):
    """Standard errors should scale approximately as 1/sqrt(m)."""
    n = 20000
    d_true = 0.2
    seed = 42

    x = arfima(n, d_true, seed=seed)

    m_values = [int(n**0.6), int(n**0.7), int(n**0.8)]
    se_values = []
    for m in m_values:
        result = estimator.estimate(x, m=m)
        se_values.append((m, result['se']))

    for i in range(len(se_values) - 1):
        # Test SE scaling (should be approximately 1/sqrt(m))
        m1, se1 = se_values[i]
        m2, se2 = se_values[i+1]
        expected_ratio = np.sqrt(m2) / np.sqrt(m1)
        actual_ratio = se1 / se2
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-1,
                                   err_msg=f"Fisher SE scaling failed: {se1} vs {se2}")


#
# Test tapered LW with 'kolmogorov' against veltaper.m baseline results
#

# Load Octave veltaper test cases
def _load_octave_veltaper_cases():
    """Load Octave veltaper results and convert to test cases."""
    json_path = os.path.join(os.path.dirname(__file__), "octave_veltaper.json")
    with open(json_path, 'r') as f:
        results = json.load(f)

    test_cases = []
    for result in results:
        test_case = {
            'name': f"{result['dataset']}:m={result['m']}",
            'dataset': result['dataset'],
            'm': result['m'],
            'expected_d_hat': result['d_hat'],
            'expected_se': result['se'],
            'expected_obj': result['obj']
        }
        test_cases.append(test_case)
    return test_cases


# Load Octave veltaper test cases
OCTAVE_VELTAPER_CASES = _load_octave_veltaper_cases()


@pytest.mark.parametrize("case", OCTAVE_VELTAPER_CASES)
def test_octave_veltaper_kolmogorov(case, nile_data, sealevel_data):
    """Test Kolmogorov taper implementation against veltaper.m results."""

    # Extract test case parameters
    dataset = case['dataset']
    m = case['m']
    expected_d_hat = case['expected_d_hat']
    expected_se = case['expected_se']
    expected_obj = case['expected_obj']

    # Get dataset from fixtures
    if dataset == 'nile':
        series = nile_data
    elif dataset == 'sealevel':
        series = sealevel_data
    else:
        pytest.skip(f"Unknown dataset: {dataset}")

    # Run Python Kolmogorov taper estimation
    lw = LW()
    result = lw.estimate(
        series,
        m=m,
        bounds=(-1.0, 3.0),  # Bounds from Octave script
        taper='kolmogorov',
        verbose=False
    )

    # Extract results
    py_d_hat = result['d_hat']
    py_se = result['se']
    py_obj = result['objective']

    # Calculate differences
    d_hat_diff = abs(py_d_hat - expected_d_hat)
    se_diff = abs(py_se - expected_se)
    obj_diff = abs(py_obj - expected_obj)

    # Print comparison for debugging (pytest with -s flag)
    print(f"\n{dataset} (m={m}):")
    print(f"  d_hat: Python={py_d_hat:10.6f}, Octave={expected_d_hat:10.6f}, diff={d_hat_diff:.2e}")
    print(f"  se:    Python={py_se:10.6f}, Octave={expected_se:10.6f}, diff={se_diff:.2e}")
    print(f"  obj:   Python={py_obj:10.6f}, Octave={expected_obj:10.6f}, diff={obj_diff:.2e}")

    # Tolerances
    d_hat_tol = 1e-4
    se_tol = 1e-8
    obj_tol = 1e-8

    # Assertions
    assert np.isfinite(py_d_hat), f"Python d_hat is not finite for {case['name']}"
    assert np.isfinite(py_se), f"Python se is not finite for {case['name']}"

    assert d_hat_diff < d_hat_tol, \
        f"d_hat mismatch for {case['name']}: Python={py_d_hat:.6f}, Octave={expected_d_hat:.6f}, diff={d_hat_diff:.2e}"
    assert se_diff < se_tol, \
        f"se mismatch for {case['name']}: Python={py_se:.6f}, Octave={expected_se:.6f}, diff={se_diff:.2e}"
    assert obj_diff < obj_tol, \
        f"obj mismatch for {case['name']}: Python={py_obj:.6f}, Octave={expected_obj:.6f}, diff={obj_diff:.2e}"

    # Check method consistency
    assert result['method'] == 'lw_velasco', f"Unexpected method: {result['method']}"
    assert result['taper'] == 'kolmogorov', f"Unexpected taper: {result['taper']}"


#
# Test against R LongMemoryTS baseline results
#

# Load R test cases
def _load_r_local_w_cases():
    """Load R LongMemoryTS baseline results and convert to test cases."""
    json_path = os.path.join(os.path.dirname(__file__), "r_local_w.json")
    with open(json_path, 'r') as f:
        r_results = json.load(f)
    test_cases = []
    for dataset, dataset_data in r_results.items():
        for size, size_data in dataset_data.items():
            for taper, case_data in size_data.items():
                test_case = {
                    'name': f"{dataset}:{size}:{taper}",
                    'dataset': dataset,
                    'size': size,
                    'taper': taper,
                    'n': case_data['n'],
                    'm': case_data['m'],
                    'expected_d_hat': case_data['d_hat'],
                    'expected_se': case_data['se'],
                    'expected_taper': case_data['taper'],
                }
                test_cases.append(test_case)
    return test_cases


# Load test cases
R_LOCAL_W_CASES = _load_r_local_w_cases()


@pytest.mark.parametrize("case", R_LOCAL_W_CASES)
def test_r_local_w_baseline(case, nile_data, sealevel_data):
    """Test local Whittle estimators against R LongMemoryTS local.W results."""

    # Extract test case parameters
    dataset = case['dataset']
    size = case['size']
    expected_d_hat = case['expected_d_hat']
    expected_se = case['expected_se']
    expected_taper = case['expected_taper']
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

    # Map taper names
    taper_map = {
        'none': 'none',
        'Velasco': 'cosine',
        'HC': 'hc'
    }
    python_taper = taper_map[expected_taper]

    # Run Local Whittle estimation
    lw = LW()
    result = lw.estimate(series, m=m, taper=python_taper, verbose=False)

    # Check basic properties
    if python_taper == 'hc':
        # HC taper reduces sample size due to differencing
        expected_n = n - 1
        assert result['n'] == expected_n, f"Sample size mismatch for {case['name']}: {result['n']} vs {expected_n} (after diff)"
    else:
        assert result['n'] == n, f"Sample size mismatch for {case['name']}: {result['n']} vs {n}"
    assert result['m'] == m, f"Bandwidth mismatch for {case['name']}: {result['m']} vs {m}"
    assert result['taper'] == python_taper, f"Taper mismatch for {case['name']}: {result['taper']} vs {python_taper}"

    # Check that results are finite
    assert np.isfinite(result['d_hat']), f"Non-finite d_hat for {case['name']}"
    assert np.isfinite(result['se']), f"Non-finite se for {case['name']}"

    # Compute differences
    d_error = abs(result['d_hat'] - expected_d_hat)
    se_error = abs(result['se'] - expected_se)

    # Print comparison for debugging (pytest with -s flag)
    print(f"\n{dataset}, taper={python_taper} (m={m}):")
    print(f"  d_hat: Python={result['d_hat']:10.6f}, R={expected_d_hat:10.6f}, diff={d_error:.2e}")
    print(f"  se:    Python={result['se']:10.6f}, R={expected_se:10.6f}, diff={se_error:.2e}")

    # Use tighter tolerance for larger sample sizes
    if size == 'small':
        atol_d = 1e-2
    elif size == 'medium':
        atol_d = 5e-3
    else:
        atol_d = 1e-3

    # Loosen tolerance for standard LW due to finite sample correction used in local.W.R
    if python_taper == 'none':
        atol_d = atol_d * 10

    # Check d_hat estimate
    assert d_error <= atol_d, \
        f"d_hat mismatch for {case['name']}: Python={result['d_hat']:.6f}, R={expected_d_hat:.6f}, error={d_error:.6f}"

    # Check standard errors
    atol_se = 1e-2
    if python_taper == 'hc':
        atol_se = atol_se * 2
    assert se_error <= atol_se, \
        f"se mismatch for {case['name']}: Python={result['se']:.6f}, R={expected_se:.6f}, error={se_error:.6f}"


def test_lw_taper_none():
    """Test result format local Whittle with no taper."""
    np.random.seed(123)
    x = np.random.randn(100)

    lw = LW()
    result = lw.estimate(x, m=20, taper='none')

    assert result['taper'] == 'none'
    assert result['method'] == 'lw'
    assert np.isfinite(result['d_hat'])
    assert np.isfinite(result['se'])


def test_lw_taper_kolmogorov():
    """Test result format local Whittle with 'kolmogorov' taper."""
    np.random.seed(456)
    x = np.random.randn(100)

    lw = LW()
    result = lw.estimate(x, m=20, taper='kolmogorov')

    assert result['taper'] == 'kolmogorov'
    assert result['method'] == 'lw_velasco'
    assert np.isfinite(result['d_hat'])
    assert np.isfinite(result['se'])


def test_lw_taper_cosine():
    """Test result format local Whittle with 'cosine' taper."""
    np.random.seed(456)
    x = np.random.randn(100)

    lw = LW()
    result = lw.estimate(x, m=20, taper='cosine')

    assert result['taper'] == 'cosine'
    assert result['method'] == 'lw_velasco'
    assert np.isfinite(result['d_hat'])
    assert np.isfinite(result['se'])


def test_lw_taper_bartlett():
    """Test result format local Whittle with 'bartlett' taper."""
    np.random.seed(456)
    x = np.random.randn(100)

    lw = LW()
    result = lw.estimate(x, m=20, taper='bartlett')

    assert result['taper'] == 'bartlett'
    assert result['method'] == 'lw_velasco'
    assert np.isfinite(result['d_hat'])
    assert np.isfinite(result['se'])


def test_lw_taper_hc():
    """Test result format local Whittle with 'hc' taper."""
    np.random.seed(789)
    x = np.random.randn(100)

    lw = LW()
    result = lw.estimate(x, m=20, taper='hc')

    assert result['taper'] == 'hc'
    assert result['method'] == 'lw_hc'
    assert np.isfinite(result['d_hat'])
    assert np.isfinite(result['se'])


def test_lw_invalid_taper():
    """Test ValueError for local Whittle with invalid taper type."""
    x = np.random.randn(50)

    lw = LW()
    with pytest.raises(ValueError, match="Unknown taper type"):
        lw.estimate(x, m=15, taper='invalid')


def test_constructor_defaults():
    """Test constructor with default parameters."""
    lw = LW()
    assert lw.bounds == (-1.0, 2.2)
    assert lw.taper == 'none'
    assert lw._default_bounds == (-1.0, 2.2)
    assert lw._default_taper == 'none'


def test_constructor_custom_params():
    """Test constructor with custom parameters."""
    lw = LW(bounds=(-0.5, 1.5), taper='hc')
    assert lw.bounds == (-0.5, 1.5)
    assert lw.taper == 'hc'


def test_repr_default_params():
    """Test __repr__ with default parameters."""
    lw = LW()
    assert repr(lw) == "LW()"


def test_repr_custom_params():
    """Test __repr__ with custom parameters."""
    lw = LW(taper='hc')
    assert repr(lw) == "LW(taper='hc')"

    lw = LW(bounds=(-0.5, 1.5), taper='kolmogorov')
    assert repr(lw) == "LW(bounds=(-0.5, 1.5), taper='kolmogorov')"


def test_fit_basic(nile_data):
    """Test basic fit functionality."""
    lw = LW()
    result = lw.fit(nile_data, m=20)

    # Should return self
    assert result is lw

    # Should have fitted attributes
    assert hasattr(lw, 'd_hat_')
    assert hasattr(lw, 'se_')
    assert hasattr(lw, 'ase_')
    assert hasattr(lw, 'n_')
    assert hasattr(lw, 'm_')
    assert hasattr(lw, 'objective_')
    assert hasattr(lw, 'nfev_')
    assert hasattr(lw, 'method_')
    assert hasattr(lw, 'taper_')
    assert hasattr(lw, 'diff_')

    # Check values are reasonable
    assert lw.n_ == len(nile_data)
    assert lw.m_ == 20
    assert np.isfinite(lw.d_hat_)
    assert lw.taper_ == 'none'
    assert lw.diff_ == 0


def test_fit_with_hc_taper(nile_data):
    """Test fit with HC taper."""
    lw = LW(taper='hc', diff=2)
    lw.fit(nile_data, m=20)

    assert lw.taper_ == 'hc'
    assert lw.diff_ == 2
    assert lw.n_ == len(nile_data) - 2  # After differencing


def test_fit_method_chaining(nile_data):
    """Test method chaining with fit."""
    d_hat = LW(taper='hc').fit(nile_data, m=20).d_hat_
    assert np.isfinite(d_hat)


def test_backward_compatibility_estimate(nile_data):
    """Test that estimate() API works."""
    lw = LW()
    result = lw.estimate(nile_data, m=20, bounds=(-0.5, 1.5), taper='hc')

    # Should return dict
    assert isinstance(result, dict)
    assert 'd_hat' in result
    assert 'se' in result
    assert result['taper'] == 'hc'
    assert 'method' in result


def test_estimate_parameter_override(nile_data):
    """Test that estimate() parameters temporarily override constructor params."""
    lw = LW(bounds=(-1.0, 2.2), taper='none')

    # Use different parameters in estimate()
    result = lw.estimate(nile_data, m=20, bounds=(-0.5, 1.5), taper='hc')

    # Constructor params should be restored
    assert lw.bounds == (-1.0, 2.2)
    assert lw.taper == 'none'


@pytest.mark.slow
def test_lw_auto_m_basic(arfima_data_auto):
    """Test that LW.fit(m='auto') works and sets bootstrap attributes."""
    x, d_true = arfima_data_auto

    lw = LW()
    lw.fit(x, m='auto')

    # Check that bootstrap attributes are set
    assert hasattr(lw, 'bootstrap_m_optimal_m_')
    assert hasattr(lw, 'bootstrap_m_iterations_')
    assert hasattr(lw, 'bootstrap_m_mse_profile_')
    assert hasattr(lw, 'bootstrap_m_k_n_')

    # Check that standard fitted attributes are set
    assert hasattr(lw, 'd_hat_')
    assert hasattr(lw, 'se_')
    assert hasattr(lw, 'm_')
    assert hasattr(lw, 'n_')

    # Check that optimal_m was used
    assert lw.m_ == lw.bootstrap_m_optimal_m_

    # Check that estimate is reasonable
    assert np.isfinite(lw.d_hat_)
    assert abs(lw.d_hat_ - d_true) < 0.3  # Loose bound

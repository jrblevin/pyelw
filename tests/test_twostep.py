import os
import json
import pytest
import numpy as np

from pyelw import TwoStepELW
from pyelw.simulate import arfima


@pytest.fixture
def estimator():
    """Create TwoStepELW estimator instance."""
    return TwoStepELW()


@pytest.mark.parametrize("d_true", [-0.8, -0.3, 0.0, 0.2, 0.4, 0.8, 1.2, 2.2, 2.8, 3.0])
@pytest.mark.parametrize("taper", ['hc', 'cosine', 'kolmogorov'])
def test_arfima(estimator, d_true, taper):
    """Test ELW estimator with various first stage tapers with ARFIMA(0,d,0) processes."""
    n = 20000
    seed = 42
    x = arfima(n, d_true, seed=seed)
    result = estimator.estimate(x, m=int(n**0.7), taper=taper, bounds=(-3, 4))

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
# Test against R LongMemoryTS baseline results
#

# Load R test cases
def _load_r_elw2s_cases():
    """Load R LongMemoryTS baseline results and convert to test cases."""
    json_path = os.path.join(os.path.dirname(__file__), "r_elw2s.json")
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
                    'trend_order': case_data['trend_order'],
                    'expected_d_hat': case_data['d_hat'],
                    'expected_se': case_data['se'],
                    'expected_taper': case_data['taper'],
                }
                test_cases.append(test_case)
    return test_cases


# Load test cases
R_ELW2s_CASES = _load_r_elw2s_cases()


@pytest.mark.parametrize("case", R_ELW2s_CASES)
def test_r_elw2s_baseline(case, nile_data, sealevel_data, estimator):
    """Test two-step exact local Whittle estimators against R LongMemoryTS ELW2S results."""

    # Extract test case parameters
    dataset = case['dataset']
    trend_order = case['trend_order']
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
        'Velasco': 'cosine',
        'HC': 'hc'
    }
    python_taper = taper_map[expected_taper]

    # Run Local Whittle estimation
    result = estimator.estimate(series, m=m, taper=python_taper, trend_order=trend_order, verbose=False)

    # Check basic properties
    assert result['n'] == n, f"Sample size mismatch for {case['name']}: {result['n']} vs {n}"
    assert result['m'] == m, f"Bandwidth mismatch for {case['name']}: {result['m']} vs {m}"
    assert result['taper'] == python_taper, f"Taper mismatch for {case['name']}: {result['taper']} vs {python_taper}"
    assert result['trend_order'] == trend_order, f"trend_order mismatch for {case['name']}: {result['trend_order']} vs {trend_order}"

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

    # Check d_hat estimate
    atol_d = 1e-4
    assert d_error <= atol_d, \
        f"d_hat mismatch for {case['name']}: Python={result['d_hat']:.6f}, R={expected_d_hat:.6f}, error={d_error:.6f}"

    # Check standard errors
    atol_se = 1e-8
    assert se_error <= atol_se, \
        f"se mismatch for {case['name']}: Python={result['se']:.6f}, R={expected_se:.6f}, error={se_error:.6f}"


def test_constructor_defaults():
    """Test constructor with default parameters."""
    elw2 = TwoStepELW()
    assert elw2.bounds == (-1.0, 2.2)
    assert elw2.taper == 'hc'
    assert elw2.trend_order == 0
    assert elw2._default_bounds == (-1.0, 2.2)
    assert elw2._default_taper == 'hc'
    assert elw2._default_trend_order == 0


def test_constructor_custom_params():
    """Test constructor with custom parameters."""
    elw2 = TwoStepELW(bounds=(-0.5, 1.5), taper='none', trend_order=1)
    assert elw2.bounds == (-0.5, 1.5)
    assert elw2.taper == 'none'
    assert elw2.trend_order == 1


def test_repr_default_params():
    """Test __repr__ with default parameters."""
    elw2 = TwoStepELW()
    assert repr(elw2) == "TwoStepELW()"


def test_repr_custom_params():
    """Test __repr__ with custom parameters."""
    elw2 = TwoStepELW(taper='none')
    assert repr(elw2) == "TwoStepELW(taper='none')"

    elw2 = TwoStepELW(trend_order=1, taper='kolmogorov')
    assert repr(elw2) == "TwoStepELW(taper='kolmogorov', trend_order=1)"


def test_fit_basic(nile_data):
    """Test basic fit functionality."""
    elw2 = TwoStepELW()
    result = elw2.fit(nile_data, m=20)

    # Should return self
    assert result is elw2

    # Should have fitted attributes
    assert hasattr(elw2, 'd_hat_')
    assert hasattr(elw2, 'se_')
    assert hasattr(elw2, 'ase_')
    assert hasattr(elw2, 'n_')
    assert hasattr(elw2, 'm_')
    assert hasattr(elw2, 'objective_')
    assert hasattr(elw2, 'nfev_')
    assert hasattr(elw2, 'd_step1_')
    assert hasattr(elw2, 'se_step1_')
    assert hasattr(elw2, 'objective_step1_')
    assert hasattr(elw2, 'nfev_step1_')

    # Check values are reasonable
    assert elw2.n_ == len(nile_data)
    assert elw2.m_ == 20
    assert np.isfinite(elw2.d_hat_)
    assert np.isfinite(elw2.d_step1_)


def test_two_stage_results(nile_data):
    """Test that two-stage results are properly stored."""
    elw2 = TwoStepELW()
    elw2.fit(nile_data, m=20)

    # Should have both stage 1 and final results
    assert np.isfinite(elw2.d_step1_)
    assert np.isfinite(elw2.d_hat_)
    assert np.isfinite(elw2.se_step1_)
    assert np.isfinite(elw2.se_)


def test_fit_method_chaining(nile_data):
    """Test method chaining with fit."""
    d_hat = TwoStepELW(taper='none').fit(nile_data, m=20).d_hat_
    assert np.isfinite(d_hat)


def test_backward_compatibility_estimate(nile_data):
    """Test that old estimate() API still works."""
    elw2 = TwoStepELW()
    result = elw2.estimate(nile_data, m=20, bounds=(-0.5, 1.5), trend_order=1)

    # Should return dict
    assert isinstance(result, dict)
    assert 'd_hat' in result
    assert 'se' in result
    assert result['method'] == '2elw'
    assert result['trend_order'] == 1
    assert 'd_step1' in result


def test_estimate_parameter_override(nile_data):
    """Test that estimate() parameters temporarily override constructor params."""
    elw2 = TwoStepELW(bounds=(-1.0, 2.2), taper='hc', trend_order=0)

    # Use different parameters in estimate()
    result = elw2.estimate(nile_data, m=20, bounds=(-0.5, 1.5), trend_order=1)

    # Constructor params should be restored
    assert elw2.bounds == (-1.0, 2.2)
    assert elw2.taper == 'hc'
    assert elw2.trend_order == 0


@pytest.mark.slow
def test_twostep_auto_m_basic(arfima_data_auto):
    """Test that TwoStepELW.fit(m='auto') works and sets bootstrap attributes."""
    x, d_true = arfima_data_auto

    ts_elw = TwoStepELW()
    ts_elw.fit(x, m='auto')

    # Check that bootstrap attributes are set
    assert hasattr(ts_elw, 'bootstrap_m_optimal_m_')
    assert hasattr(ts_elw, 'bootstrap_m_iterations_')
    assert hasattr(ts_elw, 'bootstrap_m_mse_profile_')
    assert hasattr(ts_elw, 'bootstrap_m_k_n_')

    # Check that standard fitted attributes are set
    assert hasattr(ts_elw, 'd_hat_')
    assert hasattr(ts_elw, 'se_')
    assert hasattr(ts_elw, 'm_')
    assert hasattr(ts_elw, 'n_')

    # Check that optimal_m was used
    assert ts_elw.m_ == ts_elw.bootstrap_m_optimal_m_

    # Check that estimate is reasonable
    assert np.isfinite(ts_elw.d_hat_)
    assert abs(ts_elw.d_hat_ - d_true) < 0.3  # Loose bound

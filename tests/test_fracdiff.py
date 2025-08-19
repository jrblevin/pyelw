import os
import json
import pytest  # noqa: F401
import numpy as np

from pyelw.fracdiff import fracdiff


#
# Basic Tests
#

def test_integer_differencing():
    """Test that integer differencing gives correct results."""
    x = np.array([1.0, 2.0, 4.0, 7.0, 11.0])

    # d = 0 should return identity
    result_0 = fracdiff(x, 0.0)
    np.testing.assert_array_almost_equal(result_0, x, decimal=12)

    # d = 1 should return first differences
    expected_diff1 = np.array([1.0, 1.0, 2.0, 3.0, 4.0])  # [1, 2-1, 4-2, 7-4, 11-7]
    result_1 = fracdiff(x, 1.0)
    np.testing.assert_array_almost_equal(result_1, expected_diff1, decimal=12)

    # d = -1 should return cumulative sum (inverse of differencing)
    x_simple = np.array([1.0, 1.0, 1.0, 1.0])
    expected_cumsum = np.array([1.0, 2.0, 3.0, 4.0])
    result_neg1 = fracdiff(x_simple, -1.0)
    np.testing.assert_array_almost_equal(result_neg1, expected_cumsum, decimal=12)


def test_coefficients():
    """Verify that the computed coefficients (k-d-1)/k are correct."""

    # Test with known d values
    d_values = [0.3, 0.7, 1.2, -0.4, -1.1]

    for d in d_values:
        # Generate a test sequence
        x = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Unit impulse
        result = fracdiff(x, d)

        # For unit impulse, result should be the coefficients themselves
        # Coefficient b_0 = 1, b_k = prod_{j=1}^k (j-d-1)/j for k > 0
        n = len(x)
        expected_coeffs = np.zeros(n)
        expected_coeffs[0] = 1.0

        for k in range(1, n):
            coeff = 1.0
            for j in range(1, k + 1):
                coeff *= (j - d - 1) / j
            expected_coeffs[k] = coeff

        np.testing.assert_array_almost_equal(result, expected_coeffs, decimal=12,
                                             err_msg=f"Coefficient mismatch for d={d}")

        # Verify the recursive relation: b_k = b_{k-1} * (k-d-1)/k
        for k in range(1, n):
            expected_ratio = (k - d - 1) / k
            if abs(expected_coeffs[k-1]) > 1e-15:  # Avoid division by near-zero
                actual_ratio = expected_coeffs[k] / expected_coeffs[k-1]
                assert abs(actual_ratio - expected_ratio) < 1e-12, \
                    f"Recursive relation failed at k={k}, d={d}: {actual_ratio} vs {expected_ratio}"


def test_edge_cases():
    """Test edge cases."""
    # Empty array
    empty_result = fracdiff(np.array([]), 0.5)
    assert len(empty_result) == 0

    # Single element
    single = np.array([5.0])
    single_result = fracdiff(single, 0.3)
    np.testing.assert_array_almost_equal(single_result, single, decimal=12)

    # Very large d
    x = np.array([1.0, 2.0, 3.0])
    large_d_result = fracdiff(x, 10.0)
    assert not np.isnan(large_d_result).any()
    assert np.isfinite(large_d_result).all()


#
# Test against fdiff.R from the LongMemoryTS R package
#

# Load R test cases at module level for parametrization
def _load_r_test_cases():
    """Load R-generated test cases from JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "r_fdiff.json")
    with open(json_path, 'r') as f:
        return json.load(f)


R_TEST_CASES = _load_r_test_cases()


@pytest.mark.parametrize("case", R_TEST_CASES)
def test_r_fdiff(case):
    """Test against individual R-generated test case."""
    name = case["name"]
    x = np.array(case["input"])
    d = case["d"]
    expected = np.array(case["expected"])

    # Run fracdiff
    result = fracdiff(x, d)

    # Basic shape and finiteness checks
    assert result.shape == expected.shape, f"Shape mismatch for {name}: {result.shape} vs {expected.shape}"
    assert np.isfinite(result).all(), f"Non-finite values in result for {name}"

    # Choose tolerance based on test type and magnitude
    max_val = np.max(np.abs(expected))
    if max_val < 1e-10 or "tiny" in name:
        rtol, atol = 1e-10, 1e-12
    elif max_val > 1e6 or "large" in name:
        rtol, atol = 1e-7, 1e-6
    else:
        rtol, atol = 1e-11, 1e-13

    # Main comparison
    np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol,
                               err_msg=f"R fdiff comparison failed for {name} (d={d})")


#
# Inversion tests
#

@pytest.mark.parametrize("d", [-2.1, -1.3, -0.8, -0.4, -0.1, 1e-8, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 2.1])
@pytest.mark.parametrize("n", [1, 20, 50, 100, 500])
@pytest.mark.parametrize("sigma", [1e-3, 1.0, 1e3])
def test_inversion_parametric(d, n, sigma):
    """
    Parametric test for inversion with various d, n, and scale values.

    Includes edge cases such as single element (n = 1), white noise (d=0.0),
    first differencing (d=1.0), and small d (d=1e-8), and various scale
    parameters.
    """
    np.random.seed(42)

    # Generate random data with specified scale
    u = np.random.normal(0, sigma, n)

    # Apply fractional filter (generate integrated series)
    x = fracdiff(u, -d)

    # Apply fractional differencing (recover innovations)
    u_recovered = fracdiff(x, d)

    # Check inversion quality
    mse = np.mean((u - u_recovered)**2)
    max_error = np.max(np.abs(u - u_recovered))

    # Determine tolerance scale
    u_scale = max(np.std(u), 1.0)  # Avoid division by zero for constant arrays

    # Assert inversion quality
    rtol = 1e-8
    assert mse < rtol * u_scale**2, f"MSE too high for d={d}: {mse} >= {rtol * u_scale**2}"
    assert max_error < rtol * u_scale * 10, f"Max error too high for d={d}: {max_error} >= {rtol * u_scale * 10}"


#
# Comparison against convolution-based implementation
#

def fracdiff_convolve(x: np.ndarray, d: float) -> np.ndarray:
    """
    Apply fractional differencing operator (1-L)^d to time series.

    Uses direct convolution via NumPy.  This is direct O(n^2) implementation
    used internally for testing our O(n log n) fast fracdiff implementation
    based on Jensen and Nielsen (2014).

    Parameters
    ----------
    x : np.ndarray
        Input time series
    d : float
        Fractional differencing parameter

    Returns
    -------
    np.ndarray
        Fractionally differenced series (same length as input)
    """
    n = len(x)

    if n == 0:
        return x

    # Create k vector from 1 to n-1
    k = np.arange(1, n, dtype=np.float64)

    # Compute coefficients: (k-d-1)/k
    b = (k - d - 1) / k

    # Take cumulative product and prepend 1
    b = np.concatenate([[1.0], np.cumprod(b)])

    # Apply convolution and take first n elements for causal filtering
    return np.convolve(x, b, mode='full')[:n]


def test_fracdiff_convolve_random():
    """Test that fracdiff and fracdiff_convolve produce identical outputs."""
    np.random.seed(12345)

    # Allow for differences between FFT and convolution algorithms
    rtol = 1e-10
    atol = 1e-10

    # Test various d values
    d_values = [-1.5, -0.8, -0.3, 0.0, 0.2, 0.5, 0.7, 1.0, 1.3, 1.8]

    # Test various array sizes
    sizes = [5, 8, 12, 20, 50, 100, 500]

    for d in d_values:
        for n in sizes:
            # Test with random data
            x_random = np.random.normal(0, 1, n)

            result_fast = fracdiff(x_random, d)
            result_convolve = fracdiff_convolve(x_random, d)

            # Check array shapes match
            assert result_fast.shape == result_convolve.shape, \
                f"Shape mismatch for d={d}, n={n}: {result_fast.shape} vs {result_convolve.shape}"

            # Check numerical agreement
            np.testing.assert_allclose(result_fast, result_convolve, rtol=rtol, atol=atol,
                                       err_msg=f"Mismatch for d={d}, n={n}")

            # Test with structured data
            x_linear = np.linspace(1, n, n)
            result_fast_lin = fracdiff(x_linear, d)
            result_convolve_lin = fracdiff_convolve(x_linear, d)

            np.testing.assert_allclose(result_fast_lin, result_convolve_lin, rtol=rtol, atol=atol,
                                       err_msg=f"Linear data mismatch for d={d}, n={n}")


def test_fracdiff_convolve_edge_cases():
    """Test edge cases comparing against fracdiff_convolve."""

    # Empty array
    empty = np.array([])
    assert len(fracdiff(empty, 0.5)) == len(fracdiff_convolve(empty, 0.5)) == 0

    # Single element
    single = np.array([3.14])
    result_fast = fracdiff(single, 0.7)
    result_convolve = fracdiff_convolve(single, 0.7)
    np.testing.assert_allclose(result_fast, result_convolve, rtol=1e-14, atol=1e-15)

    # Two elements
    pair = np.array([1.0, 2.0])
    for d in [-0.5, 0.0, 0.5, 1.0]:
        result_fast = fracdiff(pair, d)
        result_convolve = fracdiff_convolve(pair, d)
        np.testing.assert_allclose(result_fast, result_convolve, rtol=1e-14, atol=1e-15,
                                   err_msg=f"Two-element test failed for d={d}")

    # Test with zeros that might cause problems
    zeros_mixed = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
    for d in [-0.3, 0.0, 0.4, 1.2]:
        result_fast = fracdiff(zeros_mixed, d)
        result_convolve = fracdiff_convolve(zeros_mixed, d)
        np.testing.assert_allclose(result_fast, result_convolve, rtol=1e-12, atol=1e-14,
                                   err_msg=f"Zero-mixed test failed for d={d}")

    # Very large d (numerical stability)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    for d in [10.0, 5.0, -5.0, -10.0]:
        result_fast = fracdiff(x, d)
        result_convolve = fracdiff_convolve(x, d)

        # Both should be finite
        assert np.isfinite(result_fast).all(), f"Fast implementation gave non-finite for d={d}"
        assert np.isfinite(result_convolve).all(), f"Convolve implementation gave non-finite for d={d}"

        # Should agree numerically with looser tolerance for extreme d values
        np.testing.assert_allclose(result_fast, result_convolve, rtol=1e-10, atol=1e-12,
                                   err_msg=f"Large d test failed for d={d}")

    # Test all-zero array
    all_zeros = np.array([0.0, 0.0, 0.0, 0.0])
    for d in [-0.5, 0.0, 0.5, 1.0]:
        result_fast = fracdiff(all_zeros, d)
        result_convolve = fracdiff_convolve(all_zeros, d)

        # For all-zero input, both should give all-zero output
        assert np.allclose(result_fast, 0.0, atol=1e-15), f"All-zero fast result failed for d={d}"
        assert np.allclose(result_convolve, 0.0, atol=1e-15), f"All-zero convolve result failed for d={d}"
        np.testing.assert_allclose(result_fast, result_convolve, rtol=0, atol=1e-15,
                                   err_msg=f"All-zero comparison failed for d={d}")


@pytest.mark.parametrize("case", R_TEST_CASES)
def test_fracdiff_convolve(case):
    """Test that fracdiff_convolve produces identical outputs to fracdiff on R test cases."""
    name = case["name"]
    x = np.array(case["input"])
    d = case["d"]

    # Run both implementations
    result_fast = fracdiff(x, d)
    result_convolve = fracdiff_convolve(x, d)

    # Basic shape checks
    assert result_fast.shape == result_convolve.shape, f"Shape mismatch for {name}: {result_fast.shape} vs {result_convolve.shape}"
    assert np.isfinite(result_fast).all(), f"Non-finite values in fast result for {name}"
    assert np.isfinite(result_convolve).all(), f"Non-finite values in convolve result for {name}"

    # Choose tolerance - convolution vs FFT can have small differences
    max_val = np.max(np.abs(result_fast))
    if max_val < 1e-10 or "tiny" in name:
        rtol, atol = 1e-10, 1e-12
    elif max_val > 1e6 or "large" in name:
        rtol, atol = 1e-7, 1e-6
    else:
        rtol, atol = 1e-10, 1e-12

    # Main comparison
    np.testing.assert_allclose(result_fast, result_convolve, rtol=rtol, atol=atol,
                               err_msg=f"fracdiff vs fracdiff_convolve mismatch for {name} (d={d})")

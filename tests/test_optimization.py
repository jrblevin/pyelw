import pytest  # noqa: F401
import numpy as np

from pyelw.optimization import golden_section_search


def test_quadratic_function():
    """Test golden section search on a simple quadratic function."""
    # Minimize f(x) = (x - 2)^2, minimum at x = 2
    def quadratic(x):
        return (x - 2.0) ** 2

    bounds = (np.float64(0.0), np.float64(4.0))
    result = golden_section_search(quadratic, brack=bounds)

    assert result.success
    assert abs(result.x - 2.0) < 1e-8
    assert abs(result.fun - 0.0) < 1e-8
    assert result.nfev > 0
    assert result.nit > 0


def test_quartic_function():
    """Test on a quartic function with minimum at x = 1."""
    # Minimize f(x) = (x - 1)^4 + 0.5, minimum at x = 1
    def quartic(x):
        return (x - 1.0) ** 4 + 0.5

    bounds = (np.float64(-1.0), np.float64(3.0))
    result = golden_section_search(quartic, brack=bounds)

    assert result.success
    assert abs(result.x - 1.0) < 1e-4
    assert abs(result.fun - 0.5) < 1e-8


def test_abs_function():
    """Test on absolute value function."""
    # Minimize f(x) = |x - 1.5|, minimum at x = 1.5
    def abs_func(x):
        return abs(x - 1.5)

    result = golden_section_search(abs_func, brack=(0.0, 3.0))

    assert result.success
    assert abs(result.x - 1.5) < 1e-8
    assert abs(result.fun - 0.0) < 1e-8


def test_exp_function():
    """Test on exponential function."""
    # Minimize f(x) = e^(x-2), minimum at left boundary x = -1
    def exp_func(x):
        return np.exp(x - 2.0)

    result = golden_section_search(exp_func, brack=(-1.0, 1.0))

    assert result.success
    assert abs(result.x - (-1.0)) < 1e-6  # Should be at left boundary


def test_default_bounds():
    """Test that default bounds work correctly."""
    def simple_quad(x):
        return x ** 2

    # Should use default brack (-0.5, 1.0)
    result = golden_section_search(simple_quad)
    assert result.success

    assert abs(result.x - 0.0) < 1e-5
    assert abs(result.fun - 0.0) < 1e-10


def test_custom_tolerance():
    """Test that custom tolerance is respected."""
    def quadratic(x):
        return (x - 0.5) ** 2

    # Use loose tolerance
    result_loose = golden_section_search(quadratic, brack=(0.0, 1.0), tol=1e-3)

    # Use tight tolerance
    result_tight = golden_section_search(quadratic, brack=(0.0, 1.0), tol=1e-8)

    assert result_loose.success
    assert result_tight.success

    assert abs(result_loose.x - 0.5) < 1e-3
    assert abs(result_tight.x - 0.5) < 1e-8

    # Tight tolerance should require more iterations
    assert result_tight.nit >= result_loose.nit
    assert result_tight.nfev >= result_loose.nfev

    # Tight tolerance should be more accurate
    assert abs(result_tight.fun) < abs(result_loose.fun - 0.0)
    assert abs(result_tight.x - 0.5) < abs(result_loose.x - 0.5)


def test_maxiter_exceeded():
    """Test behavior when maximum iterations are exceeded."""
    def quadratic(x):
        return (x - 0.5) ** 2

    result = golden_section_search(quadratic, brack=(0.0, 1.0), maxiter=2)

    # Should still return a result
    assert hasattr(result, 'x')
    assert hasattr(result, 'fun')
    assert result.nit <= 2
    assert "exceeded" in result.message


def test_local_whittle_objective():
    """Test on a function similar to local Whittle objective."""
    # Simplified LW-like objective: log(mean(x^(2*d))) - 2*d*mean_log
    def lw_like_objective(d):
        # Sample periodogram values and frequencies
        I_vals = np.array([1.2, 0.8, 1.5, 0.9, 1.1])
        freqs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        weighted = I_vals * (freqs ** (2 * d))
        mean_log_freq = np.mean(np.log(freqs))

        return np.log(np.mean(weighted)) - 2 * d * mean_log_freq

    result = golden_section_search(lw_like_objective, brack=(-0.5, 1.0))

    assert result.success
    assert -0.5 <= result.x <= 1.0
    assert np.isfinite(result.fun)

    # Minimum from scipy.minimize_scalar for comparison:
    scipy_d = 0.012637697026165826
    scipy_objective = 0.09520519745285014
    assert abs(result.x - scipy_d) < 1e-6
    assert abs(result.fun - scipy_objective) < 1e-8


def test_function_evaluation_count():
    """Test that function evaluation count is accurate."""
    call_count = 0

    def counting_func(x):
        nonlocal call_count
        call_count += 1
        return (x - 0.7) ** 2

    result = golden_section_search(counting_func, brack=(0.0, 1.0))

    # nfev should match our manual count
    assert result.nfev == call_count
    assert result.nfev > 0


def test_edge_cases():
    """Test various edge cases."""
    # Function with minimum at boundary
    def boundary_min(x):
        return x

    result = golden_section_search(boundary_min, brack=(0.0, 1.0))
    assert result.success
    assert abs(result.x - 0.0) < 1e-8

    # Narrow bounds
    def narrow_func(x):
        return (x - 0.5001) ** 2

    result = golden_section_search(narrow_func, brack=(0.5, 0.501))
    assert result.success
    assert 0.5 <= result.x <= 0.501

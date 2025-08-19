import numpy as np


def fracdiff(x: np.ndarray, d: float) -> np.ndarray:
    """
    Apply fractional differencing operator (1-L)^d to time series.

    Fast fractional differencing algorithm of Jensen and Nielsen (2014).

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

    # Find next power of 2
    np2 = 1 << (2*n - 1).bit_length()

    # Single allocation for coefficients with padding
    b_full = np.zeros(np2)
    b_full[0] = 1.0

    # Compute coefficients in-place
    if n > 1:
        k = np.arange(1, n, dtype=np.float64)
        b_full[1:n] = np.cumprod((k - d - 1) / k)

    # Use rfft for real inputs
    x_fft = np.fft.rfft(x, n=np2)
    b_fft = np.fft.rfft(b_full)

    # Compute and return
    return np.fft.irfft(x_fft * b_fft, n=np2)[:n]

import numpy as np


def fracdiff(x: np.ndarray, d: float, x_fft=None) -> np.ndarray:
    """
    Apply fractional differencing operator (1-L)^d to time series.

    Fast fractional differencing algorithm of Jensen and Nielsen (2014).

    Parameters
    ----------
    x : np.ndarray
        Input time series
    d : float
        Fractional differencing parameter
    x_fft : np.ndarray, optional
        Pre-computed FFT of x. If provided, skips recomputing FFT of x.

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

    # Use cached FFT or compute it
    if x_fft is not None:
        x_fft_ = x_fft
    else:
        x_fft_ = np.fft.rfft(x, n=np2)

    # Single allocation for coefficients with padding
    b_full = np.zeros(np2)
    b_full[0] = 1.0

    # Compute coefficients in-place
    if n > 1:
        k = np.arange(1, n, dtype=np.float64)
        b_full[1:n] = np.cumprod((k - d - 1) / k)

    # Use rfft for real inputs
    b_fft = np.fft.rfft(b_full)

    # Compute and return
    return np.fft.irfft(x_fft_ * b_fft, n=np2)[:n]

import os
import pytest
import numpy as np
import pandas as pd

from pyelw.simulate import arfima


@pytest.fixture(scope="session")
def data_dir():
    """Get path to data directory."""
    return os.path.join(os.path.dirname(__file__), '..', 'data')


@pytest.fixture(scope="session")
def nile_data(data_dir):
    """Load nile time series values as NumPy array."""
    file_path = os.path.join(data_dir, "nile.csv")
    df = pd.read_csv(file_path)
    return pd.to_numeric(df['nile'], errors='coerce').values


@pytest.fixture(scope="session")
def sealevel_data(data_dir):
    """Load sealevel time series values as NumPy array."""
    file_path = os.path.join(data_dir, "sealevel.csv")
    df = pd.read_csv(file_path)
    return pd.to_numeric(df['Sea'], errors='coerce').values


@pytest.fixture
def arfima_data_auto():
    """Generate ARFIMA(0, d, 0) test data for m='auto' tests."""
    np.random.seed(42)
    d_true = 0.3
    n = 200
    x = arfima(n, d_true, sigma=1.0)
    return x, d_true

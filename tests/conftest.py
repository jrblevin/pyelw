import os
import pytest
import pandas as pd


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

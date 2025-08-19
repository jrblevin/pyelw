"""
ELW estimates from Table 8 of Shimotsu (2010) using Nelson-Plosser data.

Shimotsu, K. (2010). Exact Local Whittle Estimation of Fractional
Integration with Unknown Mean and Time Trend. _Econometric Theory_ 26,
501--540.
"""

import pandas as pd
import numpy as np
from pyelw import LW, TwoStepELW


print("Shimotsu (2010) Table 8 Replication")
print("===================================")

# Load data
data = pd.read_csv("data/nelson_plosser_ext.csv")

# Original Shimotsu (2010) results
shimotsu_original = {
    'realgnp': {'n': 80, 'lw': 1.077, 'elw': 1.126, 'ci': '[0.912, 1.340]'},
    'nomgnp': {'n': 80, 'lw': 1.273, 'elw': 1.303, 'ci': '[1.089, 1.517]'},
    'gnpperca': {'n': 80, 'lw': 1.077, 'elw': 1.128, 'ci': '[0.914, 1.342]'},
    'indprod': {'n': 129, 'lw': 0.821, 'elw': 0.850, 'ci': '[0.671, 1.029]'},
    'employmt': {'n': 99, 'lw': 0.968, 'elw': 1.000, 'ci': '[0.800, 1.200]'},
    'unemploy': {'n': 129, 'lw': 0.951, 'elw': 0.980, 'ci': '[0.801, 1.159]'},
    'gnpdefl': {'n': 100, 'lw': 1.374, 'elw': 1.398, 'ci': '[1.202, 1.594]'},
    'cpi': {'n': 129, 'lw': 1.273, 'elw': 1.287, 'ci': '[1.109, 1.466]'},
    'wages': {'n': 89, 'lw': 1.300, 'elw': 1.351, 'ci': '[1.147, 1.555]'},
    'realwag': {'n': 89, 'lw': 1.047, 'elw': 1.089, 'ci': '[0.885, 1.293]'},
    'M': {'n': 100, 'lw': 1.460, 'elw': 1.501, 'ci': '[1.305, 1.697]'},
    'velocity': {'n': 120, 'lw': 0.953, 'elw': 0.993, 'ci': '[0.808, 1.179]'},
    'interest': {'n': 89, 'lw': 1.091, 'elw': 1.108, 'ci': '[0.903, 1.312]'},
    'sp500': {'n': 118, 'lw': 0.900, 'elw': 0.958, 'ci': '[0.772, 1.143]'}
}

# Series mapping
series_mapping = [
    ('realgnp', 'Real GNP'),
    ('nomgnp', 'Nominal GNP'),
    ('gnpperca', 'Real per capita GNP'),
    ('indprod', 'Industrial production'),
    ('employmt', 'Employment'),
    ('unemploy', 'Unemployment rate'),
    ('gnpdefl', 'GNP deflator'),
    ('cpi', 'CPI'),
    ('wages', 'Nominal wage'),
    ('realwag', 'Real wage'),
    ('M', 'Money stock'),
    ('velocity', 'Velocity of money'),
    ('interest', 'Bond yield'),
    ('sp500', 'Stock prices')
]

# Initialize estimators
lw_estimator = LW()
tselw_estimator = TwoStepELW()

# Compute our results
our_results = {}

for series_code, series_label in series_mapping:
    if series_code not in data.columns:
        continue

    series_data = data[series_code].dropna().values
    # if series_code == 'unemploy':
    #     series_data = np.exp(series_data)

    n_obs = len(series_data)

    # Local Whittle: diff -> LW -> add 1
    y_diff = np.diff(series_data)
    m_lw = round(n_obs**0.70)
    lw_result = lw_estimator.estimate(y_diff, m=m_lw, verbose=False)
    d_lw = lw_result['d_hat'] + 1.0

    # Two-Step ELW
    m_elw = round(n_obs**0.70)
    elw_result = tselw_estimator.estimate(series_data, m=m_elw, detrend_order=1)
    d_elw = elw_result['d_hat']
    se_elw = elw_result['se']

    # 95% CI
    ci_lower = d_elw - 1.96 * se_elw
    ci_upper = d_elw + 1.96 * se_elw

    our_results[series_code] = {
        'n': n_obs,
        'lw': d_lw,
        'elw': d_elw,
        'ci': f'[{ci_lower:.3f}, {ci_upper:.3f}]'
    }

# Display comparison table
print(f"\n{'':<25} {'     Shimotsu (2010)':<33} {'         PyELW':<25}")
print(f"{'Series':<25} {'LW':<8} {'2ELW':<8} {'95% CI':<15} {'LW':<8} {'2ELW':<8} {'95% CI':<8}")
print("-" * 90)

for series_code, series_label in series_mapping:
    if series_code in our_results and series_code in shimotsu_original:
        orig = shimotsu_original[series_code]
        ours = our_results[series_code]
        print(f"{series_label:<25} {orig['lw']:<8.3f} {orig['elw']:<8.3f} {orig['ci']:<15} {ours['lw']:<8.3f} {ours['elw']:<8.3f} {ours['ci']:<15}")

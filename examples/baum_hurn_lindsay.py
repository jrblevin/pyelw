"""
Replication of the empirical results of Baum, Hurn, and Lindsay (2020).

The results for the nile data are on pages 576-577 and for sealevel
on pages 578-579.

Baum, C.F., S. Hurn, and K. Lindsay (2020). Local Whittle estimation
of the long-memory parameter. _Stata Journal_ 20, 565--583.
"""

import numpy as np
import pandas as pd
from pyelw import LW, ELW


def detrend(data):
    """
    Detrend by regressing on t = (1, 2, 3, 4, ...)
    """
    n = len(data)
    t = np.arange(1, n + 1)
    X = np.column_stack([np.ones(n), t])
    beta, _, _, _ = np.linalg.lstsq(X, data)
    fitted = X @ beta
    return data - fitted


# Load time series from 'nile' column of data/nile.csv
nile_df = pd.read_csv('data/nile.csv')
nile = pd.to_numeric(nile_df['nile']).values

# Load time series from 'Sea' column of data/sealevel.csv
sealevel_df = pd.read_csv('data/sealevel.csv')
sealevel = pd.to_numeric(sealevel_df['Sea']).values

# Stata always demeans the data before estimation
nile_demeaned = nile - np.mean(nile)
nile_detrended = detrend(nile)
sealevel_demeaned = sealevel - np.mean(sealevel)
sealevel_detrended = detrend(sealevel)

# Original results from Baum, Hurn, and Lindsay (2020) Stata output
original_results = {
    ('Nile', 'Demeaned', 'LW', 0.65): (0.409044, 0.06212, 0.06063),
    ('Nile', 'Detrended', 'LW', 0.65): (0.393717, 0.06541, 0.06063),
    ('Nile', 'Demeaned', 'ELW', 0.65): (0.407459, 0.06243, 0.06063),
    ('Nile', 'Detrended', 'ELW', 0.65): (0.397066, 0.06582, 0.06063),
    ('Sealevel', 'Demeaned', 'LW', 0.65): (0.859211, 0.04384, 0.04603),
    ('Sealevel', 'Demeaned', 'ELW', 0.65): (0.801612, 0.03494, 0.04603),
    ('Sealevel', 'Detrended', 'LW', 0.50): (0.551102, 0.08104, 0.08006),
    ('Sealevel', 'Detrended', 'LW', 0.65): (0.454184, 0.04165, 0.04603),
    ('Sealevel', 'Detrended', 'ELW', 0.50): (0.524068, 0.07937, 0.08006),
    ('Sealevel', 'Detrended', 'ELW', 0.65): (0.486036, 0.04394, 0.04603),
}

# Compute PyELW results
pyelw_results = {}
specs = [
    ('Nile', 'Demeaned', nile_demeaned, [0.65]),
    ('Nile', 'Detrended', nile_detrended, [0.65]),
    ('Sealevel', 'Demeaned', sealevel_demeaned, [0.50, 0.65]),
    ('Sealevel', 'Detrended', sealevel_detrended, [0.50, 0.65]),
]

for dataset, treatment, data, powers in specs:
    n = len(data)
    for alpha in powers:
        m = int(n**alpha)
        for estimator in ['LW', 'ELW']:
            if estimator == 'LW':
                lw = LW()
                result = lw.estimate(data, m=m)
            elif estimator == 'ELW':
                elw = ELW()
                result = elw.estimate(data, m=m)

            key = (dataset, treatment, estimator, alpha)
            pyelw_results[key] = (result['d_hat'], result['se'], result['ase'])

# Print comparison table
print("Baum, Hurn, and Lindsay (2020) Replication")
print("==========================================")
print()

print("NILE RIVER DATA (N=663)")
print("----------------------------------------------------------------------------------------------")
print("|           |        |       |     |      Original Results      |        PyELW Results       |")
print("| Transform | Method | Power |  m  |   d       S.E.   Asy.SE    |   d       S.E.   Asy.SE    |")
print("----------------------------------------------------------------------------------------------")

nile_specs = [
    ('Demeaned', 'LW', 0.65, 68),
    ('Demeaned', 'ELW', 0.65, 68),
    ('Detrended', 'LW', 0.65, 68),
    ('Detrended', 'ELW', 0.65, 68),
]

for treatment, method, power, m in nile_specs:
    orig_key = ('Nile', treatment, method, power)
    pyelw_key = ('Nile', treatment, method, power)

    if orig_key in original_results:
        orig_d, orig_se, orig_ase = original_results[orig_key]
        pyelw_d, pyelw_se, pyelw_ase = pyelw_results[pyelw_key]

        print(f"| {treatment:9s} | {method:6s} | {power:5.2f} | {m:3d} | "
              f"{orig_d:8.6f}  {orig_se:7.5f}  {orig_ase:7.5f} | "
              f"{pyelw_d:8.6f}  {pyelw_se:7.5f}  {pyelw_ase:7.5f} |")

print("----------------------------------------------------------------------------------------------")
print()

print("SEA LEVEL DATA (N=1558)")
print("----------------------------------------------------------------------------------------------")
print("|           |        |       |     |      Original Results      |        PyELW Results       |")
print("| Transform | Method | Power |  m  |   d       S.E.   Asy.SE    |   d       S.E.   Asy.SE    |")
print("----------------------------------------------------------------------------------------------")

sealevel_specs = [
    ('Demeaned', 'LW', 0.65, 118),
    ('Demeaned', 'ELW', 0.65, 118),
    ('Detrended', 'LW', 0.50, 39),
    ('Detrended', 'LW', 0.65, 118),
    ('Detrended', 'ELW', 0.50, 39),
    ('Detrended', 'ELW', 0.65, 118),
]

for treatment, method, power, m in sealevel_specs:
    orig_key = ('Sealevel', treatment, method, power)
    pyelw_key = ('Sealevel', treatment, method, power)

    if orig_key in original_results:
        orig_d, orig_se, orig_ase = original_results[orig_key]
        pyelw_d, pyelw_se, pyelw_ase = pyelw_results[pyelw_key]

        print(f"| {treatment:9s} | {method:6s} | {power:5.2f} | {m:3d} | "
              f"{orig_d:8.6f}  {orig_se:7.5f}  {orig_ase:7.5f} | "
              f"{pyelw_d:8.6f}  {pyelw_se:7.5f}  {pyelw_ase:7.5f} |")

print("----------------------------------------------------------------------------------------------")

#!/usr/bin/env python3
"""
Replication of tapered LW Monte Carlo in Table 2 of Shimotsu and Phillips (2005).
"""

import numpy as np

from pyelw import LW
from pyelw.simulate import arfima

# Settings
n = 500
d_list = [-3.5, -2.3, -1.7, -1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3, 1.7, 2.3, 3.5]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)

# Estimators
lw = LW()

# Initialize storage for results
hc_estimates = np.zeros((mc_reps, len(d_list)))
v_estimates = np.zeros((mc_reps, len(d_list)))

# Results matrices: (bias, se, mse) for each d value
hc_results = np.zeros((len(d_list), 3))
v_results = np.zeros((len(d_list), 3))

# Original results from Table 2 of Shimotsu and Phillips (2005)
original_hc = {
    -3.5: (2.5889, 0.3037, 6.7946),
    -2.3: (1.1100, 0.2893, 1.3157),
    -1.7: (0.4474, 0.2154, 0.2466),
    -1.3: (0.1551, 0.1231, 0.0392),
    -0.7: (0.0278, 0.0957, 0.0099),
    -0.3: (0.0100, 0.0971, 0.0095),
    0.0: (0.0034, 0.0985, 0.0097),
    0.3: (-0.0033, 0.1004, 0.0101),
    0.7: (-0.0066, 0.0994, 0.0099),
    1.3: (-0.0079, 0.0987, 0.0098),
    1.7: (0.0008, 0.0972, 0.0095),
    2.3: (0.0528, 0.0981, 0.0124),
    3.5: (-0.4079, 0.1142, 0.1795)
}

original_v = {
    -3.5: (1.6126, 0.3380, 2.7148),
    -2.3: (0.2155, 0.1726, 0.0762),
    -1.7: (0.0259, 0.1235, 0.0159),
    -1.3: (0.0081, 0.1211, 0.0147),
    -0.7: (-0.0068, 0.1219, 0.0149),
    -0.3: (-0.0133, 0.1224, 0.0151),
    0.0: (-0.0138, 0.1224, 0.0152),
    0.3: (-0.0132, 0.1235, 0.0154),
    0.7: (-0.0068, 0.1227, 0.0151),
    1.3: (0.0140, 0.1232, 0.0154),
    1.7: (0.0456, 0.1288, 0.0187),
    2.3: (-0.1781, 0.1419, 0.0519),
    3.5: (-1.4541, 0.1338, 2.1322)
}

print("Table 2 of Shimotsu and Phillips (2005)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("Original results from paper:")
print("=========================================================")
print("|     | Tapered (HC) estimator |  Tapered (V) estimator |")
print("|  d  |  bias   s.d.    MSE    |   bias   s.d.   MSE    |")
print("=========================================================")

for d_true in d_list:
    hc_orig = original_hc[d_true]
    v_orig = original_v[d_true]
    print(f"|{d_true:4.1f} | {hc_orig[0]:7.4f} {hc_orig[1]:6.4f} {hc_orig[2]:7.4f} "
          f"| {v_orig[0]:7.4f} {v_orig[1]:6.4f} {v_orig[2]:7.4f} |")

print("=========================================================")

print()
print("PyELW results:")
print("=========================================================")
print("|     | Tapered (HC) estimator |  Tapered (V) estimator |")
print("|  d  |  bias   s.d.    MSE    |   bias   s.d.   MSE    |")
print("=========================================================")

# Loop over experiments
for i, d_true in enumerate(d_list):
    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        hc_result = lw.estimate(x, m=m, taper='hc', bounds=(-4.0, 4.0), verbose=False)
        v_result = lw.estimate(x, m=m, taper='bartlett', bounds=(-4.0, 4.0), verbose=False)

        hc_estimates[rep, i] = hc_result['d_hat']
        v_estimates[rep, i] = v_result['d_hat']

    # Calculate results for each d value
    hc_results[i, 0] = np.mean(hc_estimates[:, i]) - d_true  # Bias
    hc_results[i, 1] = np.std(hc_estimates[:, i])  # S.E.
    hc_results[i, 2] = np.mean((hc_estimates[:, i] - d_true)**2)  # MSE

    v_results[i, 0] = np.mean(v_estimates[:, i]) - d_true  # Bias
    v_results[i, 1] = np.std(v_estimates[:, i])  # S.E.
    v_results[i, 2] = np.mean((v_estimates[:, i] - d_true)**2)  # MSE

    print(f"|{d_true:4.1f} | {hc_results[i, 0]:7.4f} {hc_results[i, 1]:6.4f} {hc_results[i, 2]:7.4f} "
          f"| {v_results[i, 0]:7.4f} {v_results[i, 1]:6.4f} {v_results[i, 2]:7.4f} |")

print("=========================================================")

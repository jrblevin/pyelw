#!/usr/bin/env python3
"""
Replication of ELW Monte Carlo in Table 1 of Shimotsu and Phillips (2005).
"""

import numpy as np

from pyelw import LW, ELW
from pyelw.simulate import arfima

# Settings
n = 500
d_list = [-3.5, -2.3, -1.7, -1.3, -0.7, -0.3, 0.0, 0.3, 0.7, 1.3, 1.7, 2.3, 3.5]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)

# Estimators
lw = LW()
elw = ELW()

# Initialize storage for results
elw_estimates = np.zeros((mc_reps, len(d_list)))
lw_estimates = np.zeros((mc_reps, len(d_list)))

# Results matrices: (bias, se, mse) for each d value
elw_results = np.zeros((len(d_list), 3))
lw_results = np.zeros((len(d_list), 3))

# Original results from Table 1 of Shimotsu and Phillips (2005)
original_elw = {
    -3.5: (-0.0024, 0.0787, 0.0062),
    -2.3: (-0.0020, 0.0774, 0.0060),
    -1.7: (-0.0020, 0.0776, 0.0060),
    -1.3: (-0.0014, 0.0770, 0.0059),
    -0.7: (-0.0024, 0.0787, 0.0062),
    -0.3: (-0.0033, 0.0777, 0.0060),
    0.0: (-0.0029, 0.0784, 0.0061),
    0.3: (-0.0020, 0.0782, 0.0061),
    0.7: (-0.0017, 0.0777, 0.0060),
    1.3: (-0.0014, 0.0781, 0.0061),
    1.7: (-0.0025, 0.0780, 0.0061),
    2.3: (-0.0026, 0.0772, 0.0060),
    3.5: (-0.0016, 0.0770, 0.0059)
}

original_lw = {
    -3.5: (3.1617, 0.2831, 10.076),
    -2.3: (1.6345, 0.3041, 2.7640),
    -1.7: (0.8709, 0.2788, 0.8363),
    -1.3: (0.4109, 0.2170, 0.2160),
    -0.7: (0.0353, 0.0885, 0.0091),
    -0.3: (-0.0027, 0.0781, 0.0061),
    0.0: (-0.0075, 0.0781, 0.0062),
    0.3: (-0.0066, 0.0785, 0.0062),
    0.7: (0.0099, 0.0812, 0.0067),
    1.3: (-0.2108, 0.0982, 0.0541),
    1.7: (-0.6288, 0.1331, 0.4130),
    2.3: (-1.2647, 0.1046, 1.6104),
    3.5: (-2.4919, 0.0724, 6.2150)
}

print("Table 1 of Shimotsu and Phillips (2005)")
print(f"n={n}, m=n^{{0.65}}={m}, replications={mc_reps}")
print()
print("Original results from paper:")
print("=========================================================")
print("|     |   Exact estimator      |  Untapered estimator   |")
print("|  d  |  bias   s.d.    MSE    |   bias   s.d.   MSE    |")
print("=========================================================")

for d_true in d_list:
    elw_orig = original_elw[d_true]
    lw_orig = original_lw[d_true]
    print(f"|{d_true:4.1f} | {elw_orig[0]:7.4f} {elw_orig[1]:6.4f} {elw_orig[2]:7.4f} |"
          f" {lw_orig[0]:7.4f} {lw_orig[1]:6.4f} {lw_orig[2]:7.4f} |")

print("=========================================================")


print()
print("PyELW results:")
print("=========================================================")
print("|     |   Exact estimator      |  Untapered estimator   |")
print("|  d  |  bias   s.d.    MSE    |   bias   s.d.   MSE    |")
print("=========================================================")

# Loop over experiments
for i, d_true in enumerate(d_list):
    for rep in range(mc_reps):
        x = arfima(n, d_true, sigma=1.0, seed=42 * i + rep)
        elw_result = elw.estimate(x, m=m, bounds=(-4.0, 4.0), verbose=False)
        lw_result = lw.estimate(x, m=m, bounds=(-4.0, 4.0), verbose=False)

        elw_estimates[rep, i] = elw_result['d_hat']
        lw_estimates[rep, i] = lw_result['d_hat']

    # Calculate results for each d value
    elw_results[i, 0] = np.mean(elw_estimates[:, i]) - d_true  # Bias
    elw_results[i, 1] = np.std(elw_estimates[:, i])  # S.E.
    elw_results[i, 2] = np.mean((elw_estimates[:, i] - d_true)**2)  # MSE

    lw_results[i, 0] = np.mean(lw_estimates[:, i]) - d_true  # Bias
    lw_results[i, 1] = np.std(lw_estimates[:, i])  # S.E.
    lw_results[i, 2] = np.mean((lw_estimates[:, i] - d_true)**2)  # MSE

    print(f"|{d_true:4.1f} | {elw_results[i, 0]:7.4f} {elw_results[i, 1]:6.4f} {elw_results[i, 2]:7.4f} "
          f"| {lw_results[i, 0]:7.4f} {lw_results[i, 1]:6.4f} {lw_results[i, 2]:7.4f} |")

print("=========================================================")

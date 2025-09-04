#!/usr/bin/env python3
"""
Replication of 2-step ELW Monte Carlo in Table 2 of Shimotsu (2010).

This simulation focuses on the two-step ELW estimator only, comparing
our PyELW implementation against the original published results.

Shimotsu, K. (2010). Exact Local Whittle Estimation of Fractional
Integration with Unknown Mean and Time Trend. _Econometric Theory_ 26,
501--540.
"""

import numpy as np

from pyelw import TwoStepELW
from pyelw.simulate import arfima

# Settings from Shimotsu (2010) Table 2
n = 512
d_list = [0.0, 0.4, 0.8, 1.2]
rho_list = [0.0, 0.5, 0.8]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)  # m = n^0.65 = 57

print(f"Monte Carlo replication: n={n}, m=n^{{{alpha:.2f}}}={m}, replications={mc_reps}")

# Two-step ELW estimator
elw2 = TwoStepELW()

# Original results from Shimotsu (2010) Table 2 (2ELW columns only)
original_results = {
    (0.0, 0.0): (-0.0022, 0.0058),
    (0.0, 0.5): (0.0994, 0.0061),
    (0.0, 0.8): (0.4133, 0.0072),
    (0.4, 0.0): (0.0001, 0.0058),
    (0.4, 0.5): (0.1003, 0.0060),
    (0.4, 0.8): (0.4160, 0.0072),
    (0.8, 0.0): (-0.0003, 0.0058),
    (0.8, 0.5): (0.0988, 0.0060),
    (0.8, 0.8): (0.4125, 0.0073),
    (1.2, 0.0): (-0.0006, 0.0057),
    (1.2, 0.5): (0.0990, 0.0061),
    (1.2, 0.8): (0.4117, 0.0070)
}

print("===============================================================================")
print("|     |     | Original |  PyELW   |          | Original |   PyELW  |          |")
print("|  d  | rho |   Bias   |   Bias   |   Diff   |   Var    |    Var   |   Diff   |")
print("===============================================================================")

# Loop over all parameter combinations
for d_true in d_list:
    for rho in rho_list:
        estimates = np.zeros(mc_reps)

        # Monte Carlo simulation
        for rep in range(mc_reps):
            # Generate ARFIMA(1,d,0) process
            seed = 42 * len(d_list) * len(rho_list) * rep + d_list.index(d_true) * len(rho_list) + rho_list.index(rho)
            x = arfima(n, d_true, phi=rho, sigma=1.0, seed=seed, burnin=2*n)

            # Apply 2-step ELW estimator
            result = elw2.estimate(x, m=m, bounds=(-1.0, 3.0), trend_order=0)
            estimates[rep] = result['d_hat']

        # Calculate bias and variance
        pyelw_bias = np.mean(estimates) - d_true
        pyelw_var = np.var(estimates)

        # Get original results
        orig_bias, orig_var = original_results[(d_true, rho)]

        # Calculate differences
        bias_diff = pyelw_bias - orig_bias
        var_diff = pyelw_var - orig_var

        print(f"| {d_true:3.1f} | {rho:3.1f} | {orig_bias:8.4f} | {pyelw_bias:8.4f} | {bias_diff:8.4f} |"
              f" {orig_var:8.4f} | {pyelw_var:8.4f} | {var_diff:8.4f} |")

print("===============================================================================")

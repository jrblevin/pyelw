#!/usr/bin/env python3
"""
Replication of 2-step ELW Monte Carlo in Table 3 of Shimotsu (2010).
"""

import numpy as np

from pyelw import LW, ELW, TwoStepELW
from pyelw.simulate import arfima

# Settings from Shimotsu (2010) Table 3
n = 128
d_list = [-0.4, 0.0, 0.4, 0.8, 1.0, 1.2, 1.6]
mc_reps = 10000
alpha = 0.65
m = int(n**alpha)  # m = n^0.65 = 23
bounds = (-1.0, 3.0)

print(f"Monte Carlo replication: n={n}, m=n^{{{alpha:.2f}}}={m}, replications={mc_reps}")

# Two-step ELW estimator
hc = LW(bounds=bounds, taper='hc', diff=1)
elw = ELW(bounds=bounds, mean_est='mean')
elw2 = TwoStepELW(bounds=bounds, trend_order=0)
elw2_dt = TwoStepELW(bounds=bounds, trend_order=1)

print("================================================================================================")
print("|      |         ELW         |         2ELW        |  2ELW w/Detrending  |          HC         |")
print("|   d  |   Bias   |    SD    |   Bias   |    SD    |   Bias   |    SD    |   Bias   |    SD    |")
print("================================================================================================")

# Loop over all d values
for d_true in d_list:
    est_hc = np.zeros(mc_reps)
    est_elw = np.zeros(mc_reps)
    est_elw2 = np.zeros(mc_reps)
    est_elw2_dt = np.zeros(mc_reps)

    # Monte Carlo simulation
    for rep in range(mc_reps):
        # Generate ARFIMA(1,d,0) process
        seed = 42 * len(d_list) * rep + d_list.index(d_true)
        x = arfima(n, d_true, sigma=1.0, seed=seed, burnin=2*n)

        # Apply HC (tapered) estimator
        est_hc[rep] = hc.fit(x, m=m).d_hat_

        # Apply ELW estimator with mean estimation
        est_elw[rep] = elw.fit(x, m=m).d_hat_

        # Apply 2-step ELW estimator
        est_elw2[rep] = elw2.fit(x, m=m).d_hat_

        # Apply 2-step ELW estimator with detrending
        est_elw2_dt[rep] = elw2_dt.fit(x, m=m).d_hat_

    # Compute bias and sd. for each estimate
    bias_elw = np.mean(est_elw) - d_true
    sd_elw = np.std(est_elw)

    bias_elw2 = np.mean(est_elw2) - d_true
    sd_elw2 = np.std(est_elw2)

    bias_elw2_dt = np.mean(est_elw2_dt) - d_true
    sd_elw2_dt = np.std(est_elw2_dt)

    bias_hc = np.mean(est_hc) - d_true
    sd_hc = np.std(est_hc)

    # Print results
    print(f"| {d_true:4.1f} | {bias_elw:8.4f} | {sd_elw:8.4f} | {bias_elw2:8.4f} | {sd_elw2:8.4f} | {bias_elw2_dt:8.4f} | {sd_elw2_dt:8.4f} | {bias_hc:8.4f} | {sd_hc:8.4f} |")

print("================================================================================================")

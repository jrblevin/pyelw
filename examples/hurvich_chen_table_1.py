#!/usr/bin/env python3
"""
Replication of Hurvich-Chen (2000) Monte Carlo experiments.

This script replicates the results of Table I from Hurvich and Chen (2000)
for the tapered estimator using simulated ARFIMA(1,d,0) data.

Also generates datasets for cross-platform validation with R.

Reference:

Hurvich, C. M., and W. W. Chen (2000). An Efficient Taper for Potentially
Overdifferenced Long-Memory Time Series. _Journal of Time Series Analysis_,
21, 155--180.
"""

import numpy as np
import os
import sys

from pyelw.lw import LW
from pyelw.simulate import arfima

# Number of Monte Carlo replications
N_REP = 500

# Number of observations per replication
N_OBS = 500

# Bandwidth parameter
M = 36

# Reported specifications and results from Table I
paper_results = {
    (0.0, 0.0): {'gse_mean': 0.2742, 'gse_var': 0.0403, 'gset_mean':-0.0013, 'gset_var': 0.0186},
    (0.0, 0.5): {'gse_mean': 0.2116, 'gse_var': 0.0255, 'gset_mean': 0.0574, 'gset_var': 0.0188},
    (0.0, 0.8): {'gse_mean': 0.3534, 'gse_var': 0.0154, 'gset_mean': 0.3116, 'gset_var': 0.0198},
    (0.2, 0.0): {'gse_mean': 0.3192, 'gse_var': 0.0215, 'gset_mean': 0.1994, 'gset_var': 0.0173},
    (0.2, 0.5): {'gse_mean': 0.3098, 'gse_var': 0.0133, 'gset_mean': 0.2580, 'gset_var': 0.0175},
    (0.2, 0.8): {'gse_mean': 0.5008, 'gse_var': 0.0108, 'gset_mean': 0.5112, 'gset_var': 0.0190},
    (0.4, 0.0): {'gse_mean': 0.4389, 'gse_var': 0.0130, 'gset_mean': 0.3964, 'gset_var': 0.0174},
    (0.4, 0.5): {'gse_mean': 0.4665, 'gse_var': 0.0107, 'gset_mean': 0.4551, 'gset_var': 0.0176},
    (0.4, 0.8): {'gse_mean': 0.6878, 'gse_var': 0.0106, 'gset_mean': 0.7091, 'gset_var': 0.0190},
    (0.6, 0.0): {'gse_mean': 0.6048, 'gse_var': 0.0114, 'gset_mean': 0.5949, 'gset_var': 0.0173},
    (0.6, 0.5): {'gse_mean': 0.6548, 'gse_var': 0.0111, 'gset_mean': 0.6533, 'gset_var': 0.0176},
    (0.6, 0.8): {'gse_mean': 0.8833, 'gse_var': 0.0113, 'gset_mean': 0.9079, 'gset_var': 0.0191},
    (0.8, 0.0): {'gse_mean': 0.7929, 'gse_var': 0.0107, 'gset_mean': 0.7945, 'gset_var': 0.0173},
    (0.8, 0.5): {'gse_mean': 0.8453, 'gse_var': 0.0106, 'gset_mean': 0.8535, 'gset_var': 0.0174},
    (0.8, 0.8): {'gse_mean': 1.0758, 'gse_var': 0.0110, 'gset_mean': 1.1079, 'gset_var': 0.0190},
    (1.0, 0.0): {'gse_mean': 0.9942, 'gse_var': 0.0101, 'gset_mean': 0.9895, 'gset_var': 0.0187},
    (1.0, 0.5): {'gse_mean': 1.0459, 'gse_var': 0.0104, 'gset_mean': 1.0488, 'gset_var': 0.0187},
    (1.0, 0.8): {'gse_mean': 1.2764, 'gse_var': 0.0113, 'gset_mean': 1.2999, 'gset_var': 0.0171},
    (1.2, 0.0): {'gse_mean': 1.1923, 'gse_var': 0.0101, 'gset_mean': 1.1981, 'gset_var': 0.0169},
    (1.2, 0.5): {'gse_mean': 1.2444, 'gse_var': 0.0102, 'gset_mean': 1.2553, 'gset_var': 0.0164},
    (1.2, 0.8): {'gse_mean': 1.4414, 'gse_var': 0.0048, 'gset_mean': 1.4453, 'gset_var': 0.0056},
}


def generate_mc_data(save_data=False):
    """
    Generate ARFIMA datasets for cross-platform validation.

    If save_data=True, saves datasets to data/hc/ for later analysis.
    Returns generated datasets and estimates.
    """
    # Test cases from Table I
    test_cases = [
        (0.0, 0.0), (0.0, 0.5), (0.0, 0.8),
        (0.2, 0.0), (0.2, 0.5), (0.2, 0.8),
        (0.4, 0.0), (0.4, 0.5), (0.4, 0.8),
        (0.6, 0.0), (0.6, 0.5), (0.6, 0.8),
        (0.8, 0.0), (0.8, 0.5), (0.8, 0.8),
        (1.0, 0.0), (1.0, 0.5), (1.0, 0.8),
        (1.2, 0.0), (1.2, 0.5), (1.2, 0.8),
    ]

    if save_data:
        # Create data directory
        data_dir = "data/hc"
        os.makedirs(data_dir, exist_ok=True)
        print(f"Saving ARFIMA datasets to {data_dir}/")

    lw = LW()
    results = {}

    for d_true, phi in test_cases:
        if save_data:
            print(f"Generating data for d={d_true}, phi={phi}...")

        estimates = []
        for sim in range(N_REP):
            # Generate same data as our MC study
            x = arfima(N_OBS, d_true, phi, seed=2025+sim, burnin=2*N_OBS)

            # Save to file if requested
            if save_data:
                filename = f"{data_dir}/arfima_d{d_true:03.1f}_phi{phi:03.1f}_sim{sim:03d}.dat"
                np.savetxt(filename, x, fmt='%.10f')

            # Compute estimate
            result = lw.estimate(x, m=M, taper='hc', bounds=(-0.49, 1.49), verbose=False)
            estimates.append(result['d_hat'])

        results[(d_true, phi)] = {
            'mean': np.mean(estimates),
            'var': np.var(estimates),
            'estimates': estimates
        }

    if save_data:
        print(f"Generated {len(test_cases) * N_REP} datasets in {data_dir}/")

    return results


def run_monte_carlo():
    """Run the main Monte Carlo replication study."""
    lw = LW()

    print("Hurvich and Chen (2000) Table I Replication")
    print("===========================================")
    print()
    print("- HC tapered estimator (first differences + complex taper)")
    print(f"- Sample size: n={N_OBS}")
    print(f"- Bandwidth: m={M}")
    print(f"- Replications: {N_REP}")
    print()
    print(f"| {'d':>3} | {'phi':>3} | {'Our Est.':>9} | {'HC Est.':>9} | {'Diff.':>9} | {'Our Var.':>9} | {'HC Var.':>9} | {'Diff.':>9} |")
    print("|----:|----:|----------:|----------:|----------:|----------:|----------:|----------:|")

    specifications = paper_results.keys()

    for i, (d, phi) in enumerate(specifications):
        gset_estimates = []
        for sim in range(N_REP):
            x = arfima(N_OBS, d, phi, seed=2025+sim, burnin=2*N_OBS)
            result_gset = lw.estimate(x, m=M, taper='hc', bounds=(-0.49, 1.49), verbose=False)
            gset_estimates.append(result_gset['d_hat'])

        gset_mean = np.mean(gset_estimates)
        gset_var = np.var(gset_estimates)

        paper = paper_results.get((d, phi), {})
        est_diff = gset_mean - paper['gset_mean']
        var_diff = gset_var - paper['gset_var']
        print(f"| {d:>3.1f} | {phi:>3.1f} |"
              f" {gset_mean:>9.4f} | {paper['gset_mean']:>9.4f} | {est_diff:>9.4f} |"
              f" {gset_var:>9.4f} | {paper['gset_var']:>9.4f} | {var_diff:>9.4f} |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--save-data":
        # Generate and save data for cross-platform validation
        generate_mc_data(save_data=True)
    else:
        # Run main Monte Carlo study
        run_monte_carlo()

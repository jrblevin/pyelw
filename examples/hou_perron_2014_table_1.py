#!/usr/bin/env python3
"""
Replication of Table 1 from Hou and Perron (2014).

Table 1: Bias and RMSE for a short memory process ARFIMA(alpha, d=0, 0) with RLS.

DGP: z_t = y_t + u_t
  - y_t is ARFIMA(1, d=0, 0): (1 - alpha*L) y_t = e_t, e_t ~ N(0,1)
  - u_t is random level shift (RLS) process with expected p shifts

Reference
---------
Hou, J. and Perron, P. (2014). Modified local Whittle estimator for long
memory processes in the presence of low frequency (and other)
contaminations. _Journal of Econometrics_ 182, 309--328.
"""

import numpy as np
from pyelw import LWLFC
from pyelw.simulate import arfima


def generate_rls(n, p, sigma_eta=1.0, seed=None):
    """
    Helper for generating a random level shift (RLS) process.

    Following Definition 1 in Hou and Perron (2014, p. 311):
    u_t = sum_{s=1}^t delta_{T,s} where delta_{T,t} = pi_{T,t} * eta_t
    with eta_t ~ iid N(0, sigma_eta^2) and pi_{T,t} ~ Bernoulli(p/T, 1).

    Parameters
    ----------
    n : int
        Sample size.
    p : float
        Expected number of level shifts over the entire sample
        (probability = p/n per period).
    sigma_eta : float
        Standard deviation of level shift magnitudes.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        RLS process of length n.
    """
    rng = np.random.default_rng(seed)

    # Bernoulli draws for level shift occurrences
    probs = rng.random(n)
    shifts = probs < (p / n)

    # Level shift magnitudes
    eta = rng.normal(0, sigma_eta, n)

    # Cumulative sum of shifts
    delta = shifts * eta
    u = np.cumsum(delta)

    return u


def run_monte_carlo(T, p, beta, alpha, mc_reps=500, seed=42):
    """
    Run Monte Carlo for LWLFC estimator.

    Returns bias and RMSE.
    """
    m = int(T**beta)
    d_true = 0.0

    # LWLFC with bounds specified in Hou and Perron (2014)
    lwlfc = LWLFC(bounds=(-0.99, 0.99))

    estimates = np.zeros(mc_reps)

    for rep in range(mc_reps):
        # Replication-specific seeds
        arfima_seed = seed + 2 * rep
        rls_seed = seed + 2 * rep + 1

        # Generate ARFIMA(1, 0, 0) = AR(1) process using PyELW arfima function
        # For d=0, this reduces to AR(1): (1 - phi*L) y_t = e_t
        y = arfima(T, d=0.0, phi=alpha, sigma=1.0, seed=arfima_seed)

        # Generate RLS contamination
        u = generate_rls(T, p, sigma_eta=1.0, seed=rls_seed)

        # Observed series
        z = y + u

        # Estimate
        lwlfc.fit(z, m=m)
        estimates[rep] = lwlfc.d_hat_

    bias = np.mean(estimates) - d_true
    rmse = np.sqrt(np.mean((estimates - d_true)**2))

    return bias, rmse


def print_table(results, T_list, p_list, beta_list, metric='bias'):
    """Print results table matching the paper's format."""
    # Header row 1: p values
    header1 = f"{'T':>6} |"
    for p in p_list:
        header1 += f"         p={p:<2}         |"
    print(header1)

    # Header row 2: beta values
    header2 = f"{'':>6} |"
    for p in p_list:
        header2 += "   0.6    0.7    0.8  |"
    print(header2)

    # Separator
    print("-" * len(header1))

    # Data rows
    for T in T_list:
        row = f"{T:>6} |"
        for p in p_list:
            for beta in beta_list:
                val = results[(T, p, beta)][metric]
                row += f" {val:6.3f}"
            row += " |"
        print(row)


if __name__ == "__main__":
    # Settings from the paper
    T_list = [256, 512, 1024, 2048, 4096]
    p_list = [0, 5, 10, 20]
    beta_list = [0.6, 0.7, 0.8]
    mc_reps = 500

    print("Table 1 from Hou and Perron (2014)")
    print("Bias and RMSE for short memory ARFIMA(alpha, d=0, 0) with RLS")
    print("=============================================================")
    print()
    print(f"Settings: {mc_reps} replications, bounds=[-0.99, 0.99]")
    print()

    # Panel (a): alpha = 0
    print("-----------------------------------------------------------------------------------------------------")
    print("alpha = 0")
    print("-----------------------------------------------------------------------------------------------------")

    results_alpha0 = {}
    for T in T_list:
        for p in p_list:
            for beta in beta_list:
                bias, rmse = run_monte_carlo(T, p, beta, alpha=0.0, mc_reps=mc_reps)
                results_alpha0[(T, p, beta)] = {'bias': bias, 'rmse': rmse}

    print()
    print("(a) Bias")
    print_table(results_alpha0, T_list, p_list, beta_list, metric='bias')

    print()
    print("(b) RMSE")
    print_table(results_alpha0, T_list, p_list, beta_list, metric='rmse')

    # Panel (b): alpha = 0.3
    print()
    print("-----------------------------------------------------------------------------------------------------")
    print("alpha = 0.3")
    print("-----------------------------------------------------------------------------------------------------")

    results_alpha03 = {}
    for T in T_list:
        for p in p_list:
            for beta in beta_list:
                bias, rmse = run_monte_carlo(T, p, beta, alpha=0.3, mc_reps=mc_reps)
                results_alpha03[(T, p, beta)] = {'bias': bias, 'rmse': rmse}

    print()
    print("(a) Bias")
    print_table(results_alpha03, T_list, p_list, beta_list, metric='bias')

    print()
    print("(b) RMSE")
    print_table(results_alpha03, T_list, p_list, beta_list, metric='rmse')

    # Panel (c): alpha = 0.6
    print()
    print("-----------------------------------------------------------------------------------------------------")
    print("alpha = 0.6")
    print("-----------------------------------------------------------------------------------------------------")

    results_alpha06 = {}
    for T in T_list:
        for p in p_list:
            for beta in beta_list:
                bias, rmse = run_monte_carlo(T, p, beta, alpha=0.6, mc_reps=mc_reps)
                results_alpha06[(T, p, beta)] = {'bias': bias, 'rmse': rmse}

    print()
    print("(a) Bias")
    print_table(results_alpha06, T_list, p_list, beta_list, metric='bias')

    print()
    print("(b) RMSE")
    print_table(results_alpha06, T_list, p_list, beta_list, metric='rmse')

    # Reference values from Table 1 of Hou and Perron (2014)
    paper_ref = {
        0.0: {
            # T=256
            (256, 0, 0.6): {'bias': -0.087, 'rmse': 0.191},
            (256, 0, 0.7): {'bias': -0.047, 'rmse': 0.111},
            (256, 0, 0.8): {'bias': -0.021, 'rmse': 0.078},
            (256, 5, 0.6): {'bias': -0.021, 'rmse': 0.423},
            (256, 5, 0.7): {'bias': -0.038, 'rmse': 0.291},
            (256, 5, 0.8): {'bias': 0.004, 'rmse': 0.140},
            (256, 10, 0.6): {'bias': 0.004, 'rmse': 0.546},
            (256, 10, 0.7): {'bias': -0.017, 'rmse': 0.327},
            (256, 10, 0.8): {'bias': -0.002, 'rmse': 0.162},
            (256, 20, 0.6): {'bias': -0.011, 'rmse': 0.656},
            (256, 20, 0.7): {'bias': -0.059, 'rmse': 0.421},
            (256, 20, 0.8): {'bias': 0.023, 'rmse': 0.231},
            # T=512
            (512, 0, 0.6): {'bias': -0.052, 'rmse': 0.132},
            (512, 0, 0.7): {'bias': -0.025, 'rmse': 0.075},
            (512, 0, 0.8): {'bias': -0.008, 'rmse': 0.046},
            (512, 5, 0.6): {'bias': -0.016, 'rmse': 0.353},
            (512, 5, 0.7): {'bias': -0.007, 'rmse': 0.169},
            (512, 5, 0.8): {'bias': 0.005, 'rmse': 0.082},
            (512, 10, 0.6): {'bias': -0.014, 'rmse': 0.402},
            (512, 10, 0.7): {'bias': -0.028, 'rmse': 0.197},
            (512, 10, 0.8): {'bias': -0.012, 'rmse': 0.121},
            (512, 20, 0.6): {'bias': -0.086, 'rmse': 0.575},
            (512, 20, 0.7): {'bias': -0.055, 'rmse': 0.362},
            (512, 20, 0.8): {'bias': 0.011, 'rmse': 0.104},
            (1024, 0, 0.6): {'bias': -0.037, 'rmse': 0.095},
            (1024, 0, 0.7): {'bias': -0.016, 'rmse': 0.057},
            (1024, 0, 0.8): {'bias': -0.006, 'rmse': 0.037},
            (1024, 5, 0.6): {'bias': -0.029, 'rmse': 0.262},
            (1024, 5, 0.7): {'bias': -0.001, 'rmse': 0.130},
            (1024, 5, 0.8): {'bias': 0.003, 'rmse': 0.068},
            (1024, 10, 0.6): {'bias': -0.018, 'rmse': 0.280},
            (1024, 10, 0.7): {'bias': -0.001, 'rmse': 0.147},
            (1024, 10, 0.8): {'bias': -0.004, 'rmse': 0.071},
            (1024, 20, 0.6): {'bias': 0.008, 'rmse': 0.376},
            (1024, 20, 0.7): {'bias': -0.026, 'rmse': 0.213},
            (1024, 20, 0.8): {'bias': 0.001, 'rmse': 0.078},
            (2048, 0, 0.6): {'bias': -0.013, 'rmse': 0.068},
            (2048, 0, 0.7): {'bias': -0.009, 'rmse': 0.041},
            (2048, 0, 0.8): {'bias': -0.006, 'rmse': 0.026},
            (2048, 5, 0.6): {'bias': -0.012, 'rmse': 0.215},
            (2048, 5, 0.7): {'bias': 0.001, 'rmse': 0.089},
            (2048, 5, 0.8): {'bias': 0.001, 'rmse': 0.044},
            (2048, 10, 0.6): {'bias': 0.016, 'rmse': 0.274},
            (2048, 10, 0.7): {'bias': -0.006, 'rmse': 0.108},
            (2048, 10, 0.8): {'bias': 0.005, 'rmse': 0.046},
            (2048, 20, 0.6): {'bias': 0.013, 'rmse': 0.316},
            (2048, 20, 0.7): {'bias': -0.011, 'rmse': 0.131},
            (2048, 20, 0.8): {'bias': -0.005, 'rmse': 0.052},
        },
        0.3: {
            # T=256
            (256, 0, 0.6): {'bias': -0.022, 'rmse': 0.172},
            (256, 0, 0.7): {'bias': 0.070, 'rmse': 0.118},
            (256, 0, 0.8): {'bias': 0.168, 'rmse': 0.180},
            (256, 5, 0.6): {'bias': -0.009, 'rmse': 0.371},
            (256, 5, 0.7): {'bias': 0.122, 'rmse': 0.201},
            (256, 5, 0.8): {'bias': 0.246, 'rmse': 0.265},
            (256, 10, 0.6): {'bias': 0.215, 'rmse': 0.467},
            (256, 10, 0.7): {'bias': 0.413, 'rmse': 0.269},
            (256, 10, 0.8): {'bias': 0.524, 'rmse': 0.277},
            (256, 20, 0.6): {'bias': -0.012, 'rmse': 0.524},
            (256, 20, 0.7): {'bias': 0.131, 'rmse': 0.337},
            (256, 20, 0.8): {'bias': 0.270, 'rmse': 0.311},
            # T=512
            (512, 0, 0.6): {'bias': -0.029, 'rmse': 0.132},
            (512, 0, 0.7): {'bias': 0.041, 'rmse': 0.086},
            (512, 0, 0.8): {'bias': 0.134, 'rmse': 0.142},
            (512, 5, 0.6): {'bias': -0.015, 'rmse': 0.282},
            (512, 5, 0.7): {'bias': 0.116, 'rmse': 0.179},
            (512, 5, 0.8): {'bias': 0.194, 'rmse': 0.207},
            (512, 10, 0.6): {'bias': 0.166, 'rmse': 0.354},
            (512, 10, 0.7): {'bias': 0.354, 'rmse': 0.185},
            (512, 10, 0.8): {'bias': 0.491, 'rmse': 0.235},
            (512, 20, 0.6): {'bias': 0.059, 'rmse': 0.432},
            (512, 20, 0.7): {'bias': 0.142, 'rmse': 0.241},
            (512, 20, 0.8): {'bias': 0.242, 'rmse': 0.261},
            (1024, 0, 0.6): {'bias': -0.012, 'rmse': 0.082},
            (1024, 0, 0.7): {'bias': 0.025, 'rmse': 0.058},
            (1024, 0, 0.8): {'bias': 0.118, 'rmse': 0.122},
            (1024, 5, 0.6): {'bias': -0.009, 'rmse': 0.157},
            (1024, 5, 0.7): {'bias': 0.048, 'rmse': 0.087},
            (1024, 5, 0.8): {'bias': 0.137, 'rmse': 0.142},
            (1024, 10, 0.6): {'bias': 0.123, 'rmse': 0.203},
            (1024, 10, 0.7): {'bias': 0.267, 'rmse': 0.096},
            (1024, 10, 0.8): {'bias': 0.436, 'rmse': 0.154},
            (1024, 20, 0.6): {'bias': 0.023, 'rmse': 0.276},
            (1024, 20, 0.7): {'bias': 0.053, 'rmse': 0.117},
            (1024, 20, 0.8): {'bias': 0.160, 'rmse': 0.172},
            (2048, 0, 0.6): {'bias': -0.005, 'rmse': 0.068},
            (2048, 0, 0.7): {'bias': 0.027, 'rmse': 0.049},
            (2048, 0, 0.8): {'bias': 0.092, 'rmse': 0.096},
            (2048, 5, 0.6): {'bias': -0.004, 'rmse': 0.128},
            (2048, 5, 0.7): {'bias': 0.033, 'rmse': 0.060},
            (2048, 5, 0.8): {'bias': 0.107, 'rmse': 0.111},
            (2048, 10, 0.6): {'bias': 0.072, 'rmse': 0.140},
            (2048, 10, 0.7): {'bias': 0.206, 'rmse': 0.066},
            (2048, 10, 0.8): {'bias': 0.383, 'rmse': 0.120},
            (2048, 20, 0.6): {'bias': -0.012, 'rmse': 0.221},
            (2048, 20, 0.7): {'bias': 0.040, 'rmse': 0.083},
            (2048, 20, 0.8): {'bias': 0.013, 'rmse': 0.137},
        },
        0.6: {
            # T=256
            (256, 0, 0.6): {'bias': 0.128, 'rmse': 0.194},
            (256, 0, 0.7): {'bias': 0.299, 'rmse': 0.312},
            (256, 0, 0.8): {'bias': 0.432, 'rmse': 0.437},
            (256, 5, 0.6): {'bias': 0.176, 'rmse': 0.357},
            (256, 5, 0.7): {'bias': 0.387, 'rmse': 0.407},
            (256, 5, 0.8): {'bias': 0.498, 'rmse': 0.504},
            (256, 10, 0.6): {'bias': 0.213, 'rmse': 0.385},
            (256, 10, 0.7): {'bias': 0.414, 'rmse': 0.442},
            (256, 10, 0.8): {'bias': 0.524, 'rmse': 0.530},
            (256, 20, 0.6): {'bias': 0.221, 'rmse': 0.482},
            (256, 20, 0.7): {'bias': 0.449, 'rmse': 0.493},
            (256, 20, 0.8): {'bias': 0.561, 'rmse': 0.569},
            # T=512
            (512, 0, 0.6): {'bias': 0.093, 'rmse': 0.145},
            (512, 0, 0.7): {'bias': 0.223, 'rmse': 0.233},
            (512, 0, 0.8): {'bias': 0.392, 'rmse': 0.395},
            (512, 5, 0.6): {'bias': 0.135, 'rmse': 0.237},
            (512, 5, 0.7): {'bias': 0.307, 'rmse': 0.322},
            (512, 5, 0.8): {'bias': 0.459, 'rmse': 0.463},
            (512, 10, 0.6): {'bias': 0.178, 'rmse': 0.300},
            (512, 10, 0.7): {'bias': 0.344, 'rmse': 0.370},
            (512, 10, 0.8): {'bias': 0.492, 'rmse': 0.497},
            (512, 20, 0.6): {'bias': 0.174, 'rmse': 0.340},
            (512, 20, 0.7): {'bias': 0.388, 'rmse': 0.414},
            (512, 20, 0.8): {'bias': 0.517, 'rmse': 0.522},
            (1024, 0, 0.6): {'bias': 0.052, 'rmse': 0.099},
            (1024, 0, 0.7): {'bias': 0.170, 'rmse': 0.177},
            (1024, 0, 0.8): {'bias': 0.347, 'rmse': 0.349},
            (1024, 5, 0.6): {'bias': 0.109, 'rmse': 0.190},
            (1024, 5, 0.7): {'bias': 0.250, 'rmse': 0.260},
            (1024, 5, 0.8): {'bias': 0.406, 'rmse': 0.409},
            (1024, 10, 0.6): {'bias': 0.069, 'rmse': 0.229},
            (1024, 10, 0.7): {'bias': 0.202, 'rmse': 0.278},
            (1024, 10, 0.8): {'bias': 0.380, 'rmse': 0.438},
            (1024, 20, 0.6): {'bias': 0.108, 'rmse': 0.279},
            (1024, 20, 0.7): {'bias': 0.298, 'rmse': 0.313},
            (1024, 20, 0.8): {'bias': 0.463, 'rmse': 0.466},
            (2048, 0, 0.6): {'bias': 0.020, 'rmse': 0.066},
            (2048, 0, 0.7): {'bias': 0.125, 'rmse': 0.132},
            (2048, 0, 0.8): {'bias': 0.307, 'rmse': 0.308},
            (2048, 5, 0.6): {'bias': 0.066, 'rmse': 0.136},
            (2048, 5, 0.7): {'bias': 0.186, 'rmse': 0.194},
            (2048, 5, 0.8): {'bias': 0.361, 'rmse': 0.362},
            (2048, 10, 0.6): {'bias': 0.063, 'rmse': 0.167},
            (2048, 10, 0.7): {'bias': 0.143, 'rmse': 0.216},
            (2048, 10, 0.8): {'bias': 0.330, 'rmse': 0.385},
            (2048, 20, 0.6): {'bias': 0.085, 'rmse': 0.203},
            (2048, 20, 0.7): {'bias': 0.234, 'rmse': 0.245},
            (2048, 20, 0.8): {'bias': 0.409, 'rmse': 0.411},
        },
    }

    # Compute and display differences vs paper
    print()
    print("Comparison with Hou and Perron (2014), Table 1")
    print("==============================================")

    for alpha, results in [(0.0, results_alpha0), (0.3, results_alpha03), (0.6, results_alpha06)]:
        print()
        print(f"alpha = {alpha}")
        print("-----------------------------------------------------------------------")
        print(f"{'T':>6} {'p':>4} {'beta':>5} | {'Bias':^24} | {'RMSE':^24}")
        print(f"{'':>6} {'':>4} {'':>5} | {'Paper':>7} {'Ours':>7} {'Diff':>8} |"
              f" {'Paper':>7} {'Ours':>7} {'Diff':>8}")
        print("-----------------------------------------------------------------------")

        bias_diffs = []
        rmse_diffs = []

        for T in T_list:
            for p in p_list:
                for beta in beta_list:
                    key = (T, p, beta)
                    if key in paper_ref[alpha]:
                        paper_bias = paper_ref[alpha][key]['bias']
                        paper_rmse = paper_ref[alpha][key]['rmse']
                        our_bias = results[key]['bias']
                        our_rmse = results[key]['rmse']
                        bias_diff = our_bias - paper_bias
                        rmse_diff = our_rmse - paper_rmse
                        bias_diffs.append(abs(bias_diff))
                        rmse_diffs.append(abs(rmse_diff))
                        print(f"{T:>6} {p:>4} {beta:>5.1f} | "
                              f"{paper_bias:>7.3f} {our_bias:>7.3f} {bias_diff:>+8.3f} |"
                              f" {paper_rmse:>7.3f} {our_rmse:>7.3f} {rmse_diff:>+8.3f}")

        print("-----------------------------------------------------------------------")
        print(f"{'Mean abs diff':>17} | {'':>7} {'':>7} {np.mean(bias_diffs):>8.3f} |"
              f" {'':>7} {'':>7} {np.mean(rmse_diffs):>8.3f}")
        print(f"{'Max abs diff':>17} | {'':>7} {'':>7} {np.max(bias_diffs):>8.3f} |"
              f" {'':>7} {'':>7} {np.max(rmse_diffs):>8.3f}")

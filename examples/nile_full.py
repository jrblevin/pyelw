import pandas as pd
from pyelw import LW, ELW

# Load time series from 'nile' column of data/nile.csv
df = pd.read_csv('data/nile.csv')
nile = pd.to_numeric(df['nile']).values
print("Nile Dataset")
print("============")
print(f"Number of observations: {len(nile)}")
print(f"Average value: {sum(nile) / len(nile)}")
print(f"Example values: {nile[:6]}")
print()

# Estimate d using local Whittle estimator
# Use default number of frequencies m
lw = LW().fit(nile)

# Print results
print("Local Whittle Estimation")
print("========================")
print(f"Number of observations: {lw.n_}")
print(f"Number of frequencies used: {lw.m_}")
print(f"Estimate of d: {lw.d_hat_}")
print(f"Fisher standard error: {lw.se_}")
print(f"Asymptotic standard error: {lw.ase_}")
print(f"Objective value R(d_hat): {lw.objective_}")
print(f"Function evaluation count: {lw.nfev_}")
print()

# Estimate d using exact local Whittle estimator
# Use default number of frequencies m
elw = ELW().fit(nile)

# Print results
print()
print("Exact Local Whittle Estimation")
print("==============================")
print(f"Number of observations: {elw.n_}")
print(f"Number of frequencies used: {elw.m_}")
print(f"Estimate of d: {elw.d_hat_}")
print(f"Fisher standard error: {elw.se_}")
print(f"Asymptotic standard error: {elw.ase_}")
print(f"Objective value R(d_hat): {elw.objective_}")
print(f"Function evaluation count: {elw.nfev_}")
print()

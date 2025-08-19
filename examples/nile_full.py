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
lw = LW()
result = lw.estimate(nile)

# Print results
print("Local Whittle Estimation")
print("========================")
print(f"Number of observations: {result['n']}")
print(f"Number of frequencies used: {result['m']}")
print(f"Estimate of d: {result['d_hat']}")
print(f"Fisher standard error: {result['se']}")
print(f"Asymptotic standard error: {result['ase']}")
print(f"Objective value R(d_hat): {result['objective']}")
print(f"Function evaluation count: {result['nfev']}")
print(f"Computational time: {result['timing']['total']} seconds")
print()

# Estimate d using exact local Whittle estimator
# Use default number of frequencies m
elw = ELW()
result = elw.estimate(nile)

# Print results
print()
print("Exact Local Whittle Estimation")
print("==============================")
print(f"Number of observations: {result['n']}")
print(f"Number of frequencies used: {result['m']}")
print(f"Estimate of d: {result['d_hat']}")
print(f"Fisher standard error: {result['se']}")
print(f"Asymptotic standard error: {result['ase']}")
print(f"Objective value R(d_hat): {result['objective']}")
print(f"Function evaluation count: {result['nfev']}")
print(f"Computational time: {result['timing']['total']} seconds")
print()

# Example 3 from the PyELW paper

import numpy as np
import pandas_datareader as pdr
from pyelw import TwoStepELW

# Download real GDP from FRED into a Pandas dataframe
df = pdr.get_data_fred('GDPC1', start='1950-01-01', end='2024-12-31')
gdp = df.values.flatten()
n = len(gdp)  # Number of observations
log_gdp = np.log(gdp)  # Natural logarithm
print(f"Downloaded {n} observations for U.S. real GDP")

# Two-Step ELW estimation with linear detrending
estimator = TwoStepELW()
m = int(n**0.65)  # Choose bandwidth/number of frequencies
result = estimator.estimate(log_gdp, m=m, detrend_order=1)
ci_lower = result['d_hat'] - 1.96 * result['se']
ci_upper = result['d_hat'] + 1.96 * result['se']

# Display results
print("\nTwo-Step ELW Results:")
print(f"Sample size:           {n}")
print(f"Number of frequencies: {m}")
print(f"Estimated d:           {result['d_hat']:.4f}")
print(f"Standard error:        {result['se']:.4f}")
print(f"95% CI:                [{ci_lower:.4f}, {ci_upper:.4f}]")

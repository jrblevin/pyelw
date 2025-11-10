# Example 3 from the PyELW paper

import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pyelw import TwoStepELW

# ------------------------------------------------------------------------

# Download real GDP from FRED into a Pandas dataframe
df = pdr.get_data_fred('GDPC1', start='1950-01-01', end='2024-12-31')
gdp = df.values.flatten()
n = len(gdp)  # Number of observations
log_gdp = np.log(gdp)  # Natural logarithm
print(f"Downloaded {n} observations for U.S. real GDP")

# ------------------------------------------------------------------------

# Plot original GDP series
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.subplot(3,1,1)
quarters = pd.date_range('1950Q1', periods=n, freq='QE')
plt.plot(quarters, gdp, 'b-', linewidth=0.8)
plt.ylabel('Billions of 2017 Dollars')
plt.grid(True, alpha=0.3)
plt.tick_params(labelbottom=False)  # Remove x-axis labels

# Plot log GDP series
plt.subplot(3,1,2)
plt.plot(quarters, log_gdp, 'r-', linewidth=0.8)
plt.ylabel('Logarithm')
plt.grid(True, alpha=0.3)
plt.tick_params(labelbottom=False)  # Remove x-axis labels

# Plot first differences (growth rates)
plt.subplot(3,1,3)
growth_rates = np.diff(log_gdp)
plt.plot(quarters[1:], growth_rates, 'g-', linewidth=0.8)
plt.ylabel('Growth Rate (%)')
plt.grid(True, alpha=0.3)

# Save and show the plot
plt.savefig('gdp_sample.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------

# Two-Step ELW estimation with linear detrending
estimator = TwoStepELW(trend_order=1)
m = int(n**0.65)  # Choose bandwidth/number of frequencies
estimator.fit(log_gdp, m=m)
ci_lower = estimator.d_hat_ - 1.96 * estimator.se_
ci_upper = estimator.d_hat_ + 1.96 * estimator.se_

# Display results
print("\nTwo-Step ELW Results:")
print(f"Sample size:           {n}")
print(f"Number of frequencies: {m}")
print(f"Estimated d:           {estimator.d_hat_:.4f}")
print(f"Standard error:        {estimator.se_:.4f}")
print(f"95% CI:                [{ci_lower:.4f}, {ci_upper:.4f}]")

# ------------------------------------------------------------------------

# Compare with ADF unit root test
from statsmodels.tsa.stattools import adfuller

# Perform ADF test on log GDP
adf_result = adfuller(log_gdp, autolag='AIC')
adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]

print(f"ADF statistic: {adf_statistic:.4f}")
print(f"ADF p-value:   {adf_pvalue:.4f}")
print(f"ADF conclusion: {'Unit root' if adf_pvalue > 0.05 else 'Stationary'}")

# Example 2 from the PyELW paper

import pandas as pd
from pyelw import LW, ELW

# ------------------------------------------------------------------------

# Load time series from 'nile' column of data/nile.csv
df = pd.read_csv('data/nile.csv')
nile = pd.to_numeric(df['nile']).values
print(f"Loaded {len(nile)} observations from nile.csv")

# ------------------------------------------------------------------------

# Plot the data
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['nile'])
plt.xlabel('Year')
plt.ylabel('Annual minimum level')
plt.grid(True, alpha=0.3)
plt.savefig('nile_sample.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------

# Estimate d using local Whittle estimator
lw = LW()
lw.fit(nile)
print(f"LW estimate:  {lw.d_hat_:8.3f} ({lw.se_:.3f})")

# Estimate d using exact local Whittle estimator
elw = ELW()
elw.fit(nile)
print(f"ELW estimate: {elw.d_hat_:8.3f} ({elw.se_:.3f})")

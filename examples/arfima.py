# Example 1 from the PyELW paper

from pyelw import ELW
from pyelw.simulate import arfima

# Simulation parameters
d_true = 0.4      # True memory parameter
n = 500           # Sample size
m = int(n**0.65)  # Number of frequencies

# Simulate ARFIMA(0,d,0) process
print(f"Simulating ARFIMA(0,{d_true},0) with n={n} observations...")
x = arfima(n, d_true, sigma=1.0, seed=42)

# Estimate the memory parameter via ELW
elw = ELW()
result = elw.estimate(x, m=m)

# Display results
print(f"True d:           {d_true}")
print(f"Estimated d:      {result['d_hat']:.4f}")
print(f"Standard error:   {result['se']:.4f}")
print(f"Estimation error: {abs(result['d_hat'] - d_true):.4f}")

# Example 1 from the PyELW paper

# ------------------------------------------------------------------------

from pyelw import ELW
from pyelw.simulate import arfima

# Simulation parameters
d_true = 0.4      # True memory parameter
n = 500           # Sample size

# Simulate ARFIMA(0,d,0) process
print(f"Simulating ARFIMA(0,{d_true},0) with n={n} observations...")
x = arfima(n, d_true, sigma=1.0, seed=42)
x[:10]

# ------------------------------------------------------------------------

# Plot the simulated sample path
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(range(n), x)
plt.xlabel('$t$')
plt.ylabel('$X_t$')
plt.grid(True, alpha=0.3)
plt.savefig('arfima_sample.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------

# Plot the autocorrelation function
from statsmodels.graphics.tsaplots import plot_acf
fig, ax = plt.subplots(figsize=(10, 5))
plot_acf(x, lags=60, alpha=0.05, ax=ax)
ax.set_xlabel('Lag $k$')
ax.set_ylabel('$\\rho_k$')
plt.title('')
plt.grid(True, alpha=0.3)
plt.ylim(-0.2, 1)  # Restrict to [-0.2, 1]
plt.savefig('arfima_acf.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------

# Estimate the memory parameter via ELW
elw = ELW()
m = int(n**0.65)  # Number of frequencies
result = elw.estimate(x, m=m)

# Display results
print(f"True d:           {d_true}")
print(f"Estimated d:      {result['d_hat']:.4f}")
print(f"Standard error:   {result['se']:.4f}")
print(f"Estimation error: {abs(result['d_hat'] - d_true):.4f}")

# ------------------------------------------------------------------------

# Compare different bandwidth choices
for alpha in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    m = int(n**alpha)
    result = elw.estimate(x, m=m)
    print(f"alpha={alpha:4.2f}, m={m:3d}: "
          f"{result['d_hat']:.4f} ({result['se']:.4f})")

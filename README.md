# PyELW

This is a Python library for local Whittle and Exact Local Whittle estimation
of the memory parameter of fractionally integrated time series.

## Installation

```shell
pip install pyelw
```

PyELW requires:

- Python (>= 3.13)
- NumPy (tested with 2.3.2)

You can check the latest sources with the command

```shell
git clone https://github.com/jrblevin/pyelw.git
```

### Quick Start Examples

```python
from pyelw import LW, ELW, TwoStepELW

series = load_data()   # Replace with your data loading code
n = len(series)        # Length of time series
m = n**0.65            # Number of frequencies to use

# Local Whittle (Robinson, 1995)
lw = LW()
result = lw.estimate(series, m=m)
print(f"d_LW = {result['d_hat']}")

# Local Whittle with Hurvich-Chen taper
hc = LW(taper='hc')
result = hc.estimate(series, m=m)
print(f"d_HC = {result['d_hat']}")

# Exact local Whittle (Shimotsu and Phillips, 2005)
elw = ELW()
result = elw.estimate(series, m=m)
print(f"d_ELW = {result['d_hat']}")

# Two step ELW (Shimotsu, 2010)
elw2s = TwoStepELW()
result = elw2s.estimate(series, m=m, trend_order=1)
print(f"d_2ELW = {result['d_hat']}")
```

## Citing the Package and Methods

The recommended practice is to cite both the specific method used and the PyELW
package.  For example:

> We use the exact local Whittle estimator of Shimotsu and Phillips (2005)
> implemented in the PyELW package (Blevins, 2025).

See the references section below for full citations for each of the methods and
the PyELW package.  Here is a BibTeX entry for the PyELW paper:

```bibtex
@TechReport{pyelw,
    title = {{PyELW}: Exact Local {Whittle} Estimation for Long Memory Time Series in Python},
    author = {Jason R. Blevins},
    institution = {The Ohio State University},
    year = 2025,
    type = {Working Paper}
}
```

## Methods Implemented

- `LW` - Untapered and tapered local Whittle estimators
    - Untapered local Whittle estimator of Robinson (1995)
      (`taper='none'`, default)
    - Tapered local Whittle estimators of Velasco (1999)
      (`taper='kolmogorov'`, `taper='cosine'`, or `taper='bartlett'`)
    - Complex tapered local Whittle estimator of Hurvich and Chen (2000)
      (`taper='hc'`)
- `ELW` - Exact local Whittle estimator of Shimotsu and Phillips (2005).
- `TwoStepELW` - Two-step exact local Whittle estimator of Shimotsu (2010).

Each of these classes provides an `estimate()` method which requires
the data (a NumPy ndarray) and the number of frequencies to use.
See the PyELW paper or the examples below for details.

### LW taper Options

By default the `LW` estimator implements the standard (untapered) estimator of
Robinson (1995).  However, it also supports several taper options.

You can specify the taper at initialization:

```python
from pyelw import LW
import numpy as np

# Sample data
x = np.random.randn(500)

# Standard untapered local Whittle (Robinson, 1995) - default
lw = LW()
result = lw.estimate(x)

# Kolmogorov taper (Velasco, 1999)
lw_kol = LW(taper='kolmogorov')
result = lw_kol.estimate(x)

# Cosine bell taper (Velasco, 1999) 
lw_cos = LW(taper='cosine')
result = lw_cos.estimate(x)

# Triangular Bartlett window taper (Velasco, 1999)
lw_bart = LW(taper='bartlett')
result = lw_bart.estimate(x)

# Hurvich-Chen complex taper (Hurvich and Chen, 2000)
# Note: diff parameter specifies number of times to difference the data
lw_hc = LW(taper='hc')
result = lw_hc.estimate(x, diff=1)
```

You can also override the taper on a per-call basis:

```python
# LW instance with default taper
lw = LW(taper='hc')

# Use the default taper
result = lw.estimate(x)

# Override with a different taper for this call
result = lw.estimate(x, taper='none')
```

### Helper Functions

The library also includes the following helper functions which may be useful:

- `fracdiff` - Fast O(n log n) fractional differencing, following Jensen and Nielsen (2014).
- `arfima` - Simulation of ARFIMA(1,d,0) processes, including ARFIMA(0,d,0) as a special case.

#### Fractional Differencing

```python
from pyelw.fracdiff import fracdiff
import numpy as np

# Generate sample data
x = np.random.randn(100)

# Apply fractional differencing with d=0.3
dx = fracdiff(x, 0.3)
```

#### ARFIMA Simulation

```python
from pyelw.simulate import arfima

# Simulate ARFIMA(1,0.4,0) with phi=0.5
data = arfima(n=1000, d=0.4, phi=0.5, sigma=1.0, seed=123)
```

## Examples

### Example 1: Nile River Level Data

The following example uses Pandas to load a CSV dataset containing
observations on the level of the Nile river and estimates d via LW and ELW.

```python
import pandas as pd
from pyelw import LW, ELW

# Load time series from 'nile' column of data/nile.csv
df = pd.read_csv('data/nile.csv')
nile = pd.to_numeric(df['nile']).values
print(f"Loaded {len(nile)} observations")

# Estimate d using local Whittle estimator
# Use default number of frequencies m
lw = LW()
result = lw.estimate(nile)
print(f"LW estimate of d: {result['d_hat']}")

# Estimate d using exact local Whittle estimator
# Use default number of frequencies m
elw = ELW()
result = elw.estimate(nile)

# Print results
print(f"ELW estimate of d: {result['d_hat']}")
```

Output:

```
Loaded 663 observations
LW estimate of d: 0.40904431255796164
ELW estimate of d: 0.8859171843580865
```

### Example 2: ARFIMA(0,d,0) Process

Here we simulate an ARFIMA(0, 0.3, 0) process and use the simulated data to
estimate d via ELW.

```python
from pyelw import ELW
from pyelw.simulate import arfima

# Set simulation parameters
n = 5000          # Sample size
d_true = 0.3      # True memory parameter
sigma = 1.0       # Innovation standard deviation
seed = 42         # Random seed

# Simulate ARFIMA(0,d,0) process
print(f"Simulating ARFIMA(0,{d_true},0) with n={n} observations...")
x = arfima(n, d_true, sigma=sigma, seed=seed)

# Initialize ELW estimator
elw = ELW()

# Estimate the memory parameter
# Use m = n^0.65 frequencies
m = int(n**0.65)
result = elw.estimate(x, m=m)

# Display results
print(f"True d:         {d_true}")
print(f"Estimated d:    {result['d_hat']:.4f}")
print(f"Standard error: {result['se']:.4f}")
print(f"Estimation error: {abs(result['d_hat'] - d_true):.4f}")

# 95% confidence interval
ci_lower = result['d_hat'] - 1.96 * result['se']
ci_upper = result['d_hat'] + 1.96 * result['se']
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

Output:
```
Simulating ARFIMA(0,0.3,0) with n=5000 observations...
True d:         0.3
Estimated d:    0.3315
Standard error: 0.0318
Estimation error: 0.0315
95% CI: [0.2692, 0.3939]
```

### Example 3: Real GDP Data from FRED

Here we download real GDP data from FRED using `pandas_datareader` and
estimate d via Two Step ELW:

```python
import numpy as np
import pandas_datareader as pdr
from pyelw import TwoStepELW

# Download real GDP from FRED
print("Downloading real GDP data from FRED...")
series = pdr.get_data_fred('GDPC1', start='1950-01-01', end='2024-12-31')
gdp_data = series.dropna()
gdp = gdp_data.values.flatten()
print(f"Downloaded {len(gdp)} observations")

# Take natural logarithm for growth rate interpretation
log_gdp = np.log(gdp)
print("Using log(real GDP) for analysis")

# Initialize Two-Step ELW estimator
estimator = TwoStepELW()

# Choose bandwidth/number of frequencies
n = len(log_gdp)
m = int(n**0.65)

# Estimate d via Two-Step ELW
print("\nEstimating long memory parameter...")
print(f"Sample size: {n}")
print(f"Number of frequencies: {m}")
result = estimator.estimate(log_gdp, m=m, trend_order=1, verbose=True)

# Display results
print("\nTwo-Step ELW Results:")
print(f"Estimated d:    {result['d_hat']:.4f}")
print(f"Standard error: {result['se']:.4f}")
ci_lower = result['d_hat'] - 1.96 * result['se']
ci_upper = result['d_hat'] + 1.96 * result['se']
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

Output:

```
Downloading real GDP data from FRED...
Downloaded 300 observations
Using log(real GDP) for analysis

Estimating long memory parameter...
Sample size: 300
Number of frequencies: 40
Detrending with polynomial order 1
Using 40 frequencies for both stages
Stage 1: hc tapered LW estimation
  Stage 1 estimate: d = 1.0427
Stage 2: Exact local whittle estimation
    Starting from Stage 1: d = 1.042677
    Final estimate: d = 1.0096

Two-Step ELW Results:
Estimated d:    1.0096
Standard error: 0.0791
95% CI: [0.8547, 1.1646]
```

## Summary of Included Replications

| Filename                            | Paper                        | Reference    | Estimators              | Description                                            |
|-------------------------------------|------------------------------|--------------|-------------------------|--------------------------------------------------------|
| `hurvich_chen_table_1.py`           | Hurvich and Chen (2000)      | Table I      | `LW('hc')`              | Monte Carlo with simulated ARFIMA(1,d,0) data.         |
| `hurvich_chen_table_1.R`            | Hurvich and Chen (2000)      | Table I      | `LW('hc')`              | R version of above, demonstrating corrected code.      |
| `hurvich_chen_table_3.py`           | Hurvich and Chen (2000)      | Table III    | `LW('hc')`              | Application to IMF International Financial Statistics. |
| `shimotsu_phillips_2005_table_1.py` | Shimotsu and Phillips (2005) | Table 1      | `LW`, `ELW`             | Monte Carlo with LW and ELW with ARFIMA(1,d,0) data    |
| `shimotsu_phillips_2005_table_2.py` | Shimotsu and Phillips (2005) | Table 2      | `LW('hc', 'bartlett')`  | Monte Carlo with tapered LW estimators                 |
| `shimotsu_2010_table_2.py`          | Shimotsu (2010)              | Table 2      | `TwoStepELW`            | ELW Monte Carlo with ARFIMA(1,d,0) data.               |
| `shimotsu_2010_table_8.py`          | Shimotsu (2010)              | Table 8      | `TwoStepELW`            | Application to extended Nelson and Plosser data.       |
| `baum_hurn_lindsay.py`              | Baum, Hurn, Lindsay (2020)   | pp. 576-579  | `LW`, `ELW`             | Application to Nile river and sea level data.          |

## Unit Tests

A `pytest` comprehensive unit test suite with over 2,400 parametric tests is
included.  To run the tests, you'll need to first install the additional test
dependencies, then run `pytest`:

```shell
pip install -r requirements-test.txt
pytest
```

Note that some tests, particularly the bootstrap MSE bandwidth selection tests,
take several minutes to run.  These tests are marked as `@pytest.mark.slow`
and can be excluded with:

```bash
pytest -m "not slow"
```

## References

* Blevins, J.R. (2025).
  [PyELW: Exact Local Whittle Estimation for Long Memory Time Series in Python](https://jblevins.org/research/pyelw).
  Working Paper, The Ohio State University.

* Hurvich, C. M., and W. W. Chen (2000). An Efficient Taper for Potentially
  Overdifferenced Long-Memory Time Series. _Journal of Time Series Analysis_
  21, 155--180.

* Robinson, P. M. (1995). Gaussian Semiparametric Estimation of Long
  Range Dependence. _Annals of Statistics_ 23, 1630--1661.

* Shimotsu, K. (2010). Exact Local Whittle Estimation of Fractional
  Integration with Unknown Mean and Time Trend. _Econometric Theory_ 26,
  501--540.

* Shimotsu, K. and Phillips, P.C.B. (2005). Exact Local Whittle Estimation
  of Fractional Integration. _Annals of Statistics_ 33, 1890--1933.

* Velasco, C. (1999). Gaussian Semiparametric Estimation for Non-Stationary
  Time Series. _Journal of Time Series Analysis_ 20, 87--126.

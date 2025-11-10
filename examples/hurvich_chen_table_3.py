#!/usr/bin/env python3
"""
This script replicates the empirical results of Table III
in Hurvich and Chen (2000).

Reference:

Hurvich, C. M., and W. W. Chen (2000). An Efficient Taper for Potentially
Overdifferenced Long-Memory Time Series. _Journal of Time Series Analysis_,
21, 155--180.
"""

import numpy as np

from pyelw import LW

specifications = [
    # Description,                            filename,              transform,     m, d_hat, se
    ('Global temperatures',                   'glotemp.dat',         None,        130,  0.45, 0.060),
    ('S&P 500 Stock Index',                   'snp500.dat',          None,       1383,  0.99, 0.018),
    ('Inflation, USA',                        'cpi_us.dat',          'diff-log',   40,  0.57, 0.123),
    ('Inflation, UK',                         'cpi_uk.dat',          'diff-log',   40,  0.33, 0.123),
    ('Inflation, France',                     'cpi_fr.dat',          'diff-log',   40,  0.67, 0.123),
    ('Real wages, USA',                       'realwage_us.dat',     None,         35,  1.43, 0.121),
    ('Industrial production, USA',            'indpro_us.dat',       'log',       100,  1.34, 0.075),
]

lw = LW(taper='hc')

print("Hurvich and Chen (2000) Table III Replication")
print("=============================================")
print()
print(f"| {'Dataset':<40} | {'n':>4} | {'m':>4} | {'Our Est.    ':>16} | {'HC Est.    ':>16} |")
print("|:-----------------------------------------|-----:|-----:|:----------------:|:----------------:|")

for desc, filename, transform, m, paper_d_hat, paper_se, in specifications:
    series = np.loadtxt(f"data/{filename}")
    n = len(series) - 1  # Report n - 1 for differencing
    if transform == 'log':
        series = np.log(series)
    elif transform == 'diff-log':
        series = np.diff(np.log(series))
    lw.fit(series, m=m)
    our_d_hat = lw.d_hat_
    our_se = lw.se_
    print(f"| {desc:<40} | {n:>4d} | {m:>4d} | {our_d_hat:>7.2f} ({our_se:>5.3f})  | {paper_d_hat:>7.2f} ({paper_se:>5.3f})  | ")

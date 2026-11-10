# Assignment 5 — Two-stock PDE and Monte Carlo

Student ID: **83019cd** · Group 2 · 20 points

The contract pays `max(S1(T), S2(T))`. Both stocks start at 100, follow
independent GBMs, have equal constant volatility and pay no dividends.
The brief leaves volatility, maturity and rate unspecified; the main example
uses 20%, one year and a 5% annual rate. Other parameter cases are included.

- `two_stock.py`: analytic price, full two-dimensional finite difference solver,
  and exact-terminal Monte Carlo with estimated standard error and 95% CI.
- `assignment_5.ipynb`: analytic derivation, Itô/hedging PDE derivation, finite
  difference construction, executed convergence studies and comparisons.
- `test_two_stock.py`: independent integration benchmark, pricing identities,
  numerical refinement, Monte Carlo sampling and invalid-input checks.
- `figures/`: exported PDE convergence, price surface and Monte Carlo plots.
- `*_convergence.csv`, `parameter_comparison.csv`: experiment results.

## Analytic result

The payoff is one stock plus an exchange option. For equal initial spots of
100 and common volatility sigma, its value is
`200 * Phi(sigma * sqrt(T / 2))`, with Phi the standard normal CDF.
For the main example this is approximately **111.24629160**.
Interest rate cancels under these no-dividend, common-drift assumptions.

## Numerical methods

The PDE solver discretizes both log-stock coordinates with central differences
and uses Crank–Nicolson with two initial backward-Euler half steps to smooth
the terminal kink. A sparse LU factorization is reused. Requested spots lie
on grid nodes; `space_steps` must be even.

At the artificial edges of the truncated log domain, the solver uses the
approximation `V = max(S1,S2)`. This is not exact near equal-stock corners.
The notebook separately checks domain expansion at fixed mesh spacing,
spatial refinement and time refinement. Prices near the artificial edges
should not be treated as accurate merely because the central price converges.
Neither boundary values nor numerical methods call the analytic pricer.

Monte Carlo uses two independent exact GBM terminal draws per simulation,
so there is no time discretization error. Discounting is combined with the
exponential using payoff homogeneity. Standard errors use the unbiased sample
variance and independent simulations. Confidence intervals are approximate
CLT intervals, not guaranteed bounds.

Both numerical functions return result dataclasses with prices, grid or
uncertainty information, and runtime. `analytic_price`, `finite_difference_price`
and `monte_carlo_price` accept `spot1`, `spot2`, `volatility`, `maturity`, `rate`.
PDE settings additionally include `space_steps`, `time_steps`, `log_half_width`;
MC settings include `num_paths` and `seed`. See docstrings and the notebook.

## Run

From repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_5 -q
```

Open the notebook using the course Python environment and run all cells.
The notebook locates the repository root whether launched from the root or
this homework folder. Keep all files together. No external market data,
network access or additional dependencies are required.

References: course stochastic calculus and BSM notes; spring 2026 finite
method introduction (same two-stock project), BSPDE and Monte Carlo notebooks.
Full repository paths and relevant concepts are listed inside the notebook.

# Group 1 — Assignment 2: Tree calibration

Implements the 15-point balanced binomial tree assignment in the final cell of
`Lectures/2026_fall_BME/04_Tree/Tree.ipynb`.

## Submit these four files together

- `assignment_2.ipynb`: executed examples, formulas, numerical results and three figures.
- `tree_calibration.py`: lecture tree helpers, balanced call pricer and root solver.
- `test_tree_calibration.py`: 55 dedicated correctness tests.
- `README.md`: source, usage, requirements and submission notes.

Student ID: `83019cd`.
Submission folder: `homeworks/83019cd/assignment_2`.
Submission branch: `20261020_83019cd_homework`.
The supplied deadline is 21 October 2026.

## Sources and conventions

Tree helper names and construction follow
`Lectures/2026_fall_BME/04_Tree/Binomial_tree_coding.ipynb`:
`european_call_payoff`, `create_spot_tree`, `create_discounted_price_tree`,
and `calcBalancedDownStep`.

For n steps over T years and constant continuously compounded annual rate r:

- `df_step = exp(-r*T/n)`; total discount is `df_step**n = exp(-r*T)`.
- `growth = 1/df_step`; `down = 2*growth - up`.
- Risk-neutral up/down probabilities are exactly 0.5 each.
- The up-step returned is a multiplier: up=1.2 means a 20% up move per step.

`calibrate(observed_price, steps, K, expiry, rate, spot)` uses
`scipy.optimize.brentq` to solve tree price minus observed price equals zero.
The lecture suggests `minimize`; brentq directly solves the required equation.
The search is bounded to positive-stock balanced trees. Infeasible targets,
non-identifiable deterministic-minimum targets, and zero-down boundary targets
raise descriptive errors.

The general lecture pricer is corrected to normalize first-level stock prices
by the initial spot and apply down/up probabilities to their respective child
nodes. Its stored prices are discounted to time zero, as in the lecture.
The balanced pricer uses the equivalent conventional backward recursion,
discounting at each step with fixed probabilities of 0.5. No extra discounting
is applied to the root price.

Do not substitute the library's volatility-based `BalancedSimpleBinomialTree`:
its factors differ from the exactly 50/50 construction requested by this lecture.
No lecture or `src` files were modified. Assignment 1 is not a dependency.

## Running

Use the existing course Python environment and root `requirements.txt`.
From the repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_2 -q
```

Open `assignment_2.ipynb` in Jupyter with the course environment and run all
cells in order. Keep the Python module beside the notebook.

Example:

```python
from tree_calibration import calibrate, balanced_call_price

up = calibrate(observed_price=10.450583572185565, steps=8,
               K=100., expiry=1., rate=.05, spot=100.)
price = balanced_call_price(spot=100., spot_mult_up=up, steps=8,
                            K=100., expiry=1., rate=.05)
# up is approximately 1.076632; price matches the observed price.
```

## Requirement review and results

- Numerical root solver: bracketed Brent method.
- Same contract: calibrated with 1, 2, 4, 8, 16, 32, 64 and 128 steps.
- Flat rate: one-step discount factors scaled by expiry/number of steps.
- One-step contracts: strikes 80, 90, 100, 110, 120 and expiries .25, 1, 2 years.
- Plots: step-count comparison and residuals; factors against strike by expiry;
  six annotated one-step tree diagrams.
- Synthetic observed prices come from the BSM formula in the local BME BSM
  lecture, with volatility 20% for the common contract and 25% for the one-step
  contract examples. The calibrator itself does not require a BSM model or vol.
- Maximum absolute price residual: 1.492e-13 for the common contract across
  step counts and 1.066e-14 across the one-step contract examples.
- Verification: 55 pytest tests, including independent binomial expectations,
  one-step closed-form calibration, recovered known factors, negative rates,
  correct probability ordering, discount scaling and invalid/unattainable inputs.

Each contract in the second demonstration is fitted separately. This is not a
single tree fitted jointly to all strikes. Assumptions: European calls on a
non-dividend-paying stock, positive spot/strike/expiry, integer steps, flat rate,
and positive calibrated down multipliers. Calibration precision is numerical;
matching a contract price is not a claim about accuracy for other contracts.

# Assignment 2 — Tree calibration

The task is in the final cell of `Lectures/2026_fall_BME/04_Tree/Tree.ipynb`.
The tree helper functions come from `Binomial_tree_coding.ipynb` in that folder.

- `tree_calibration.py` contains the tree functions and `calibrate`.
- `assignment_2.ipynb` compares step counts and one-step contracts, with plots.
- `test_tree_calibration.py` checks prices, recovered step sizes and discounting.

For n steps over T years, the per-step discount factor is `exp(-r*T/n)`.
The balanced tree has probabilities 0.5 and down multiplier `2/df_step - up`.
`calibrate(observed_price, steps, K, expiry, rate, spot)` uses
`scipy.optimize.brentq` to find the up multiplier. For example, up=1.2 means
a 20% up move. The notebook explains the price bounds used to bracket the root.

The examples use Black–Scholes prices as synthetic observations. The same
contract is fitted with 1–128 steps. Separate one-step trees are then fitted
for different strikes and expiries, all at the same annual interest rate.

Two corrections were needed in the general lecture pricer: divide first-level
stock prices by the initial spot to obtain multipliers, and weight the lower
child by the down probability. The tests include spot values other than one
and unequal probabilities to check these corrections.

Run the tests from the repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_2 -q
```

Open the notebook with the course Python environment and run the cells in order.
Keep all four files together.

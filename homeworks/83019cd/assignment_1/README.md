# Group 1 — Assignment 1

Covers 1(a), 1(b), 1(c), and 1(d): 15 available points, subject to grading.

## Files to submit together

- `assignment_1.ipynb`: executed explanation, formulas, examples and plots.
- `option_basics.py`: plotting customization, call price/Greeks and put price.
- `test_option_basics.py`: pytest verification.
- `README.md`: usage and submission notes.

The starting functions are from the 2026 BME BSM lecture notebook, cells 9 and 43.
The separate `src` pricing library and lecture files are unchanged.

## Running

Use the course environment and dependencies in the root `requirements.txt`.
From the repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_1 -q
```

Open `assignment_1.ipynb` in Jupyter with the course environment, and run all
cells in order. Keep `option_basics.py` beside the notebook.

## Verification

61 homework tests pass. The notebook checks 180 parameter combinations
(including negative rates); maximum absolute parity residual is 2.487e-14.
The tolerance is 1e-12. Reference prices at S0=K=100, T=1, r=0.05,
sigma=0.2 are call 10.4505835722 and put 5.5735260223.
All five analytic call Greeks are checked against central finite differences.

Greeks use decimal rates and volatility: vega and rho are per unit change;
divide by 100 for a percentage-point move. Theta is per year of calendar time
passing. The supported pricing domain is T>0, sigma>0, S0>0 and K>0,
with finite scalar r, T, S0 and sigma and scalar or one-dimensional strikes.

## Submission placement

Student ID: `83019cd`.
Folder: `homeworks/83019cd/assignment_1`.
Branch: `20261020_83019cd_homework`.
Official deadline supplied for this assignment: 21 October 2026.

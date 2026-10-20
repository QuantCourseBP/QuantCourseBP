# Assignment 3 — Digital options

The starting files were copied from
`Lectures/2026_fall_BME/05_BuildingBlocks/solution`, as requested in the lecture.

`EuropeanDigitalContract` was added to `contract.py`, and
`EuropeanDigitalAnalyticPricer` to `pricer.py`. The original market model,
market data and vanilla pricers are retained. Local imports use relative paths
so this library copy can be tested alongside the other assignments.

The digital pays one unit at expiry: the call when the stock is above the
strike, and the put when it is below. Both pay zero exactly at the strike.
The pricer calculates fair value, delta, gamma, vega, theta and rho under the
same no-dividend BSM assumptions as the lecture. Pricing requires positive
remaining time and volatility; use `payoff` for an expiry observation.

`assignment_3.ipynb` derives the formulas and compares normalized bull spreads
`[C(K)-C(K+h)]/h` with the digital call. It shows both the payoff and price
convergence as h decreases. References are included in the notebook.

`test_pricer.py` keeps the lecture's 24 parity tests. `test_digital.py` adds
checks for digital fair value, Greeks, payoff conventions and spread convergence.

Run the tests from the repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_3 -q
```

Run the notebook with the course Python environment. Keep all nine files,
including `__init__.py`, together in this folder. The shared enums are imported
from the repository's `src.enums`.

# Assignment 1 — Option basics and Black–Scholes

Starting point: `my_plotter` and `black_scholes_eur_call` in
`Lectures/2026_fall_BME/03_bsm/presentation_03_option_basics_and_bsm_model_2026.ipynb`.

- `option_basics.py` contains the extended plotter, call pricer with Greeks,
  and put pricer.
- `assignment_1.ipynb` contains the formulas, examples and parity plots.
- `test_option_basics.py` checks prices, Greeks, parity and plot settings.

The call pricer still returns a price array by default. Set
`return_greeks=True` to return a dictionary with the price and five Greeks.
The lecture's reference to the global `K_vec` was replaced with the argument `K`.

Prices assume a non-dividend-paying stock and constant rate and volatility.
Spot, strike, remaining time and volatility must be positive. Vega and rho
are per unit input change; divide by 100 for a percentage point. Theta is
per year of calendar time passing.

Run the tests from the repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_1 -q
```

Open the notebook with the course Python environment and run the cells in order.
Keep all four files together.

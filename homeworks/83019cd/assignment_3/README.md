# Group 1 — Assignment 3: Building the Pricing Library

Covers 3(a), digital options (12 points), and 3(b), the bull-spread limit
(8 points). Points remain subject to grading.

## Professor's starting-point instruction

The `Assignments - Digital options` cell in
`Lectures/2026_fall_BME/05_BuildingBlocks/notebook_building_blocks.ipynb`
says to create a folder under `homeworks` and copy this lecture's `solution`
contents. This folder starts with all five files from
`Lectures/2026_fall_BME/05_BuildingBlocks/solution`.

Changes to that copy:

- `contract.py`: adds `EuropeanDigitalContract`, directly extending `Contract`.
- `pricer.py`: adds `EuropeanDigitalAnalyticPricer`, extending `Pricer` and
  reusing the copied vanilla pricer's d1/d2 helpers.
- `model.py`: local import made relative; model behavior unchanged.
- `market_data.py`: exact unmodified copy.
- `test_pricer.py`: local import made relative; all 24 lecture parity cases
  retained without changing assertions.
- `__init__.py`: makes this library copy a package to avoid collisions with
  similarly named modules in other lectures and assignments.
- `test_digital.py`: 104 new digital-specific test cases.
- `assignment_3.ipynb`: executed explanation, Greek check and convergence figure.
- `README.md`: these source, usage and submission notes.

Original lectures, the repository's `src` library and Assignments 1/2 were not
modified. The copy shares `src.enums`, as the lecture solution does, and must
remain inside the repository. No external pricing package is required.

## Model and conventions

The digital is cash-or-nothing with a unit cash payout:

- Call payoff: 1 if terminal spot is strictly above strike, otherwise 0.
- Put payoff: 1 if terminal spot is strictly below strike, otherwise 0.
- Exactly at strike both pay zero; this event has probability zero in the
  continuous BSM model with positive time and volatility.
- Short positions reverse all payoffs, prices and Greeks.

Use the lecture's constructor convention:

```python
EuropeanDigitalContract(underlying, derivative_type, long_short, strike, expiry)
EuropeanDigitalAnalyticPricer(contract, model, Params())
```

The pricer implements `calc_fair_value`, `calc_delta`, `calc_gamma`, `calc_vega`,
`calc_theta`, and `calc_rho`, with no arguments, matching the copied lecture's
interface. It uses `model.spot`, `model.vol`, `model.risk_free_rate`, and
`expiry - valuation_time`. Assumptions: European exercise, no dividends, flat
continuously compounded rate, positive spot, strike, volatility and remaining
time. For an expiry payoff use the contract's `payoff` method rather than
pre-expiry Greek formulas. The observation dictionary uses the rounded expiry
returned by `get_timeline`, as in the lecture.

Call price is `exp(-r*tau)*N(d2)`; put price is `exp(-r*tau)*N(-d2)`.
The notebook derives these from the risk-neutral terminal stock distribution
and derives all five Greeks by differentiation. Digital formulas are an
extension of the lecture's BSM material, not a claim that the digital formulas
were printed in the lecture. Vega/rho are per unit input change (divide by 100
for a percentage point). Theta is per year of calendar time passing.

Supplementary primary references, also linked in the notebook:

- [QuantLib cash-or-nothing payoff implementation](https://github.com/lballabio/QuantLib/blob/master/ql/instruments/payoffs.cpp).
- [Elsevier publisher sample, Pricing Theory, printed page 54](https://booksite.elsevier.com/samplechapters/9780120476824/9780120476824.PDF#page=54), for scaled-spread limits.

External reference code was not copied. The digital formulas and code follow
the derivation documented in the notebook.

## Bull-spread demonstration

Using the actual copied vanilla and new digital pricers, the notebook compares
`[C(K) - C(K+h)] / h` with the digital call price for strike gaps 10, 5, 1 and
0.1. The division by h normalizes the payoff to one unit. The figure shows
terminal payoffs, pre-expiry prices across initial stock prices, and maximum
absolute price error against strike gap.

For spot=strike=100, expiry one year, rate 5%, vol 20%:

- Digital call price: 0.5323248155.
- Digital put price: 0.4189046090.
- Normalized spread price at h=0.1: 0.5313867947.
- Maximum spread/digital price error over the spot grid falls from about
  9.128e-2 at h=10 to 9.484e-4 at h=0.1.

The limit is pointwise at expiry, not uniform near the payoff discontinuity.
Very small strike gaps can magnify floating point cancellation.

## Running and verification

From the repository root with the course environment:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_3 -q
```

All 128 Assignment 3 cases pass (24 copied professor cases and 104 new cases).
New tests cover reference prices, independent quadrature of terminal indicator
payoffs, all five Greeks against finite differences, long/short signs, digital
call + put = discounted unit cash and its Greek identities, payoff/timeline
conventions, invalid inputs and price convergence of normalized bull spreads.
The three homework folders together pass 244 tests.

Open `assignment_3.ipynb` in Jupyter with the course environment and run its
cells in order. It includes executed numerical results and an embedded figure.
The leading repository hash printed on import is existing `src/__init__.py`
behavior. Local library imports use the `assignment_3` package name; keep this
folder name if relocating it under your personal submission folder.

## Files and final submission placement

Submit all nine files listed above together. Do not submit `__pycache__` or
`.pytest_cache` directories.

Student ID: `83019cd`.
Submission folder: `homeworks/83019cd/assignment_3`.
Submission branch: `20261020_83019cd_homework`.
The supplied Group 1 deadline is 21 October 2026.

# Assignment 4 — Volatility surface and numerical Greeks

Student ID: **83019cd** · Group 2 · 15 points

Starting point: `Lectures/2026_fall_BME/06_FlatVol/solution`, with the lecture's
`contract.py` and `enums.py`. Local imports and the market-data path are adapted
for this folder. The shared library and earlier assignments are unchanged.

- `pricer.py`: forward, backward and central delta/gamma with absolute or relative bumps.
- `enums.py`: `FiniteDifferenceMethod` and `BumpType` labels.
- `assignment_4.ipynb`: formulas, assumptions, executed examples, error analysis and recommendations.
- `test_numerical_greeks.py`: accuracy, stencils, parity, convergence and state preservation tests.
- `figures/`: exported error plots.
- `greek_errors.csv`, `optimal_bumps.csv`: reproducible measurements and best sampled settings.

The required settings are class variables on `Pricer`:
`finite_difference_method` and `bump_type`. Instances can override them.
`absolute_bump_size` is a currency increment; `relative_bump_size` is a fraction
of the original spot. A relative size of `0.0001` means 0.01%, not 0.0001%.
Both conventions use a fixed effective increment throughout each stencil.

Example (after importing this package's classes):

```python
MarketData.initialize()
model = FlatVolModel(Stock.BLUECHIP_BANK)
contract = EuropeanContract(model.underlying, PutCallFwd.CALL,
                            LongShort.LONG, model.spot, 1.0)
pricer = EuropeanAnalyticPricer(contract, model, Params())
pricer.finite_difference_method = FiniteDifferenceMethod.CENTRAL
pricer.bump_type = BumpType.RELATIVE
pricer.relative_bump_size = 1e-4
print(pricer.calc_delta(GreekMethod.BUMP))
print(pricer.calc_gamma(GreekMethod.BUMP))
```

The lecture's central relative 1% default is retained for compatibility, not as
an optimal accuracy recommendation. See the notebook for separately calibrated
delta and gamma bump sizes. Conclusions apply to the tested option set and
floating-point implementation; there is no universal optimal bump.

Prices use the repository's `mkt` data and 5% annual rate, with no dividends.
FlatVol holds strike/expiry volatility fixed during spot bumps. All stencil
spots must be positive. The implementation preserves valuation time, instance
settings and original model state by copying the pricer before revaluation.
Only delta and gamma are extended; other Greeks retain the lecture code.

Run from repository root:

```sh
venv_quant_course_v312/bin/python -m pytest homeworks/83019cd/assignment_4 -q
```

Open the notebook with the course environment and run all cells in order.
Keep all files together; market inputs remain in the repository's `mkt` folder.

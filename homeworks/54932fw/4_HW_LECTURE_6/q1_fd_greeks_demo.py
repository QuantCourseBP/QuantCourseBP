# Homework 4: numerical delta and gamma with every finite difference method and bump type (running example).
# Run from this folder with the course venv:  python q1_fd_greeks_demo.py   (printed output: q1_fd_greeks_demo.out)
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))                                   # the library of this folder
repo_root = next(p for p in [here, *here.parents] if (p / 'mkt' / 'spot.csv').exists())

from market_data import MarketData
MarketData.market_folder = repo_root / 'mkt'                    # csv files of the repository (Lecture 6)
MarketData.initialize()
from pricer import *                                            # library of Lecture 6 + NEW VERSION of Pricer

# running example: ATM 1Y call on BLUECHIP_BANK, FlatVol model (Lecture 6, cell 22)
und = Stock.BLUECHIP_BANK
spot = MarketData.get_spot()[und]
call = EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, spot, 1.0)
pricer = EuropeanAnalyticPricer(call, FlatVolModel(und), Params())
delta, gamma = pricer.calc_delta(GreekMethod.ANALYTIC), pricer.calc_gamma(GreekMethod.ANALYTIC)
print(f'ATM call: S = K = {spot}, T = 1, vol = {pricer.model.get_vol(spot, 1.0):.6f}, '
      f'value = {pricer.calc_fair_value():.6f}')
print(f'analytic: delta = {delta:.6f}, gamma = {gamma:.6f}')

# 1. default class variables = the lecture's choice (central, relative 1%): same numbers as the lecture
print(f'\ndefault: {Pricer.fd_method.value}, {Pricer.bump_type.value}, '
      f'relative_bump_size = {Pricer.relative_bump_size}')
d_bump, g_bump = pricer.calc_delta(GreekMethod.BUMP), pricer.calc_gamma(GreekMethod.BUMP)
print(f'bump: delta = {d_bump:.6f}, gamma = {g_bump:.6f}   (lecture: 0.626132, 0.007642)')
assert round(d_bump, 6) == 0.626132 and round(g_bump, 6) == 0.007642

# 2. the six combinations: relative 1% (h = 1.3017) and absolute h = 1
print(f"\n{'method':9s}{'bump':9s}{'h':>8s}{'delta':>11s}{'error':>11s}{'gamma':>11s}{'error':>11s}")
results = {}
for bump_type in BumpType:
    for fd_method in FDMethod:
        Pricer.fd_method, Pricer.bump_type = fd_method, bump_type     # class variables: used by every pricer
        h = pricer.calc_spot_bump_size()
        d, g = pricer.calc_delta(GreekMethod.BUMP), pricer.calc_gamma(GreekMethod.BUMP)
        results[fd_method, bump_type] = (d, g)
        print(f'{fd_method.value:9s}{bump_type.value:9s}{h:8.4f}{d:11.6f}{d - delta:11.2e}{g:11.6f}{g - gamma:11.2e}')
Pricer.fd_method, Pricer.bump_type = FDMethod.CENTRAL, BumpType.RELATIVE  # back to the default

# 3. checks: (forward + backward) / 2 = central for delta; forward and backward delta errors ~ +- h/2 * gamma
for bump_type in BumpType:
    fwd, bwd, cen = (results[m, bump_type][0] for m in FDMethod)
    h = Pricer.relative_bump_size * spot if bump_type == BumpType.RELATIVE else Pricer.absolute_bump_size
    print(f'{bump_type.value:9s}(fwd + bwd)/2 - central = {(fwd + bwd) / 2 - cen: .1e};  '
          f'fwd error = {fwd - delta:.6f}, h/2 * gamma = {h / 2 * gamma:.6f}')
    assert abs((fwd + bwd) / 2 - cen) < 1e-12

# 4. one pricer only: an instance attribute hides the class variable
pricer.fd_method = FDMethod.FORWARD
other = EuropeanAnalyticPricer(call, FlatVolModel(und), Params())
print(f'\ninstance: {pricer.fd_method.value}, other pricer: {other.fd_method.value}, class: {Pricer.fd_method.value}')
del pricer.fd_method

# 5. a wrong label is refused
Pricer.fd_method = 'SIDEWAYS'
try:
    pricer.calc_delta(GreekMethod.BUMP)
except ValueError as error:
    print('error:', error)
Pricer.fd_method = FDMethod.CENTRAL

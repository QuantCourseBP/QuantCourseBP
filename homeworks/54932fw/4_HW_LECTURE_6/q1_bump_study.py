# Homework 4: error of the numerical delta and gamma against the bump size, for every finite difference method;
# absolute against relative bumping on a set of European options; suggested (optimal) bump sizes.
# Run from this folder with the course venv:  python q1_bump_study.py   (output: q1_bump_study.out, figures/)
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))                                   # the library of this folder
repo_root = next(p for p in [here, *here.parents] if (p / 'mkt' / 'spot.csv').exists())

import matplotlib
matplotlib.use('Agg')                                           # files only (no window)
import matplotlib.pyplot as plt
import numpy as np
from market_data import MarketData
MarketData.market_folder = repo_root / 'mkt'
MarketData.initialize()
from pricer import *                                            # library of Lecture 6 + NEW VERSION of Pricer

(here / 'figures').mkdir(exist_ok=True)
sizes = 10.0 ** -np.arange(1.0, 9.01, 0.5)                      # relative bumps eps = 1e-1, ..., 1e-9 (17 values)
abs_sizes = 10.0 ** -np.arange(-1.0, 7.01, 0.5)                 # absolute bumps h = 1e1, ..., 1e-7 (17 values)
greeks = ('delta', 'gamma')


def set_bump(fd_method: FDMethod, bump_type: BumpType, size: float) -> None:
    Pricer.fd_method, Pricer.bump_type = fd_method, bump_type   # class variables: used by every pricer
    if bump_type == BumpType.RELATIVE:
        Pricer.relative_bump_size = size
    else:
        Pricer.absolute_bump_size = size


def rel_error(pricer: Pricer, greek: str) -> float:
    exact = getattr(pricer, f'calc_{greek}')(GreekMethod.ANALYTIC)
    return abs(getattr(pricer, f'calc_{greek}')(GreekMethod.BUMP) - exact) / abs(exact)


# 1. ATM 1Y call on BLUECHIP_BANK (running example): relative error against eps, 3 methods
und = Stock.BLUECHIP_BANK
spot = MarketData.get_spot()[und]
atm = EuropeanAnalyticPricer(EuropeanContract(und, PutCallFwd.CALL, LongShort.LONG, spot, 1.0),
                             FlatVolModel(und), Params())
atm_err = {}
for fd_method in FDMethod:
    for greek in greeks:
        errs = []
        for eps in sizes:
            set_bump(fd_method, BumpType.RELATIVE, eps)
            errs.append(rel_error(atm, greek))
        atm_err[fd_method, greek] = np.array(errs)
print('1. ATM call, relative bump: relative error of delta | gamma')
print('eps     ' + ''.join(f'{m.value:>10s}' for m in FDMethod) + ' |' + ''.join(f'{m.value:>10s}' for m in FDMethod))
for i, eps in enumerate(sizes):
    print(f'{eps:7.0e} ' + ''.join(f'{atm_err[m, "delta"][i]:10.1e}' for m in FDMethod) + ' |'
          + ''.join(f'{atm_err[m, "gamma"][i]:10.1e}' for m in FDMethod))
for greek in greeks:                                            # slope of log(error) on eps in [1e-3, 1e-1]
    slopes = [np.polyfit(np.log10(sizes[:5]), np.log10(atm_err[m, greek][:5]), 1)[0] for m in FDMethod]
    print(f'slopes {greek}: ' + ', '.join(f'{m.value} {s:.2f}' for m, s in zip(FDMethod, slopes)))

# 2. the set of European options: 4 stocks x CALL, PUT x K/S = 0.9, 1, 1.1 x T = 0.5, 1, 2 (72 options)
book = [EuropeanAnalyticPricer(EuropeanContract(s, cp, LongShort.LONG, k * MarketData.get_spot()[s], t),
                               FlatVolModel(s), Params())
        for s in Stock for cp in (PutCallFwd.CALL, PutCallFwd.PUT) for k in (0.9, 1.0, 1.1) for t in (0.5, 1.0, 2.0)]
worst = {}                                                      # worst relative error over the options of a stock
for bump_type, grid in ((BumpType.RELATIVE, sizes), (BumpType.ABSOLUTE, abs_sizes)):
    for fd_method in FDMethod:
        for greek in greeks:
            table = np.zeros((len(Stock), len(grid)))
            for j, size in enumerate(grid):
                set_bump(fd_method, bump_type, size)
                for i, stock in enumerate(Stock):
                    table[i, j] = max(rel_error(p, greek) for p in book if p.contract.underlying == stock)
            worst[bump_type, fd_method, greek] = table
set_bump(FDMethod.CENTRAL, BumpType.RELATIVE, 0.01)             # back to the lecture's default
Pricer.absolute_bump_size = 1.0

# 3. suggested bump sizes: the grid value with the smallest worst error over all 72 options
print(f'\n2. {len(book)} options; suggested bumps (worst relative error over the set at that bump)')
print(f"{'greek':6s}{'method':9s}{'relative eps*':>14s}{'error':>9s}{'absolute h*':>13s}{'error':>9s}"
      + ''.join(f'{"h* " + s.value[:8]:>13s}' for s in Stock))
for greek in greeks:
    for fd_method in FDMethod:
        rel = worst[BumpType.RELATIVE, fd_method, greek].max(axis=0)
        ab = worst[BumpType.ABSOLUTE, fd_method, greek]
        per_stock = [abs_sizes[int(np.argmin(row))] for row in ab]
        print(f'{greek:6s}{fd_method.value:9s}{sizes[int(np.argmin(rel))]:14.0e}{rel.min():9.1e}'
              f'{abs_sizes[int(np.argmin(ab.max(axis=0)))]:13.0e}{ab.max(axis=0).min():9.1e}'
              + ''.join(f'{h:13.0e}' for h in per_stock))
print('spots: ' + ', '.join(f'{s.value} {MarketData.get_spot()[s]}' for s in Stock))
for greek in greeks:                                            # the relative eps* does not depend on the stock
    rows = worst[BumpType.RELATIVE, FDMethod.CENTRAL, greek]
    print(f'central {greek}, relative eps* per stock: ' + ', '.join(f'{sizes[int(np.argmin(r))]:.0e}' for r in rows))

# one relative bump for both greeks (the library has one relative_bump_size): central differences
both = np.maximum(worst[BumpType.RELATIVE, FDMethod.CENTRAL, 'delta'].max(axis=0),
                  worst[BumpType.RELATIVE, FDMethod.CENTRAL, 'gamma'].max(axis=0))
best = int(np.argmin(both))
lecture = int(np.argmin(abs(sizes - 0.01)))
print(f'central, one eps for delta and gamma: eps* = {sizes[best]:.0e}, worst error {both[best]:.1e}  '
      f'(lecture eps = 1e-02: {both[lecture]:.1e})')

# 4. data for the figures of the report (csv) and the figures (png)
np.savetxt(here / 'figures' / 'atm_errors.csv',
           np.column_stack([sizes] + [atm_err[m, g] for g in greeks for m in FDMethod]), delimiter=',',
           header='eps,' + ','.join(f'{g}_{m.value}' for g in greeks for m in FDMethod), comments='')
for bump_type, grid in ((BumpType.RELATIVE, sizes), (BumpType.ABSOLUTE, abs_sizes)):
    for greek in greeks:
        np.savetxt(here / 'figures' / f'worst_{greek}_{bump_type.value.lower()}_central.csv',
                   np.column_stack([grid, worst[bump_type, FDMethod.CENTRAL, greek].T]), delimiter=',',
                   header='size,' + ','.join(s.value for s in Stock), comments='')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, greek in zip(axes, greeks):
    for m, mark in zip(FDMethod, 'o^s'):
        ax.loglog(sizes, atm_err[m, greek], marker=mark, label=m.value.lower())
    ax.set_title(f'{greek}: ATM 1Y call, BLUECHIP_BANK, relative bump')
    ax.set_xlabel('relative bump eps (h = eps * S)')
    ax.set_ylabel('relative error')
    ax.grid(True, which='major', ls=':')
    ax.legend()
fig.tight_layout()
fig.savefig(here / 'figures' / 'q1_error_vs_bump.png', dpi=150)
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for row, greek in enumerate(greeks):
    for col, (bump_type, grid) in enumerate(((BumpType.ABSOLUTE, abs_sizes), (BumpType.RELATIVE, sizes))):
        ax = axes[row, col]
        for i, s in enumerate(Stock):
            ax.loglog(grid, worst[bump_type, FDMethod.CENTRAL, greek][i], marker='o', ms=3,
                      label=f'{s.value} (S = {MarketData.get_spot()[s]})')
        ax.set_title(f'central {greek}, {bump_type.value.lower()} bump: worst error per stock')
        ax.set_xlabel('h' if bump_type == BumpType.ABSOLUTE else 'eps (h = eps * S)')
        ax.set_ylabel('worst relative error')
        ax.grid(True, which='major', ls=':')
        ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(here / 'figures' / 'q1_absolute_vs_relative.png', dpi=150)
print('figures saved in figures/')

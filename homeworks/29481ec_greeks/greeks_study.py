"""
Homework: finite-difference delta/gamma in Pricer - analytic vs bump, absolute vs relative bump.
Run from the repository root:   python homeworks/29481ec_greeks/greeks_study.py
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pricer import *

MarketData.initialize()
OUT = os.path.dirname(os.path.abspath(__file__))

METHODS = list(FiniteDifferenceMethod)
BUMPS = list(BumpType)
STOCKS = [Stock.BLUECHIP_BANK, Stock.TIPTOP_SOLUTIONS, Stock.MONEY_MALL]
MONEYNESS = [0.8, 1.0, 1.2]            # strike / spot
EXPIRIES = [0.25, 1.0, 2.0]
KINDS = [PutCallFwd.CALL, PutCallFwd.PUT]
REL_GRID = np.logspace(-9, -0.5, 60)   # relative bump sizes (fraction of spot)
ABS_GRID = np.logspace(-7, 1.5, 60)    # absolute bump sizes (currency units), the SAME for every option


def make_pricer(stock, kind, k_over_s, expiry):
    model = BSVolModel(stock)
    contract = EuropeanContract(stock, kind, LongShort.LONG, k_over_s * model.spot, expiry)
    return EuropeanAnalyticPricer(contract, model, Params())


def bump_greeks(pricer, fd, bump_type, size):
    Pricer.fd_method, Pricer.bump_type, Pricer.spot_bump_size = fd, bump_type, size
    return pricer.calc_delta(GreekMethod.BUMP), pricer.calc_gamma(GreekMethod.BUMP)


def run_study():
    rows = []
    for stock, kind, m, T in itertools.product(STOCKS, KINDS, MONEYNESS, EXPIRIES):
        pr = make_pricer(stock, kind, m, T)
        spot = pr.model.spot
        d_an, g_an = pr.calc_delta(GreekMethod.ANALYTIC), pr.calc_gamma(GreekMethod.ANALYTIC)
        for fd, bt in itertools.product(METHODS, BUMPS):
            for size in (REL_GRID if bt == BumpType.RELATIVE else ABS_GRID):
                d, g = bump_greeks(pr, fd, bt, size)
                rel = size if bt == BumpType.RELATIVE else size / spot   # bump as a fraction of spot
                rows.append(dict(stock=stock.value, kind=kind.value, moneyness=m, expiry=T, spot=spot,
                             fd=fd.value, bump_type=bt.value, grid=size, rel=rel, size=size * (spot if bt == BumpType.RELATIVE else 1),
                             delta_err=abs(d - d_an), gamma_err=abs(g - g_an) / abs(g_an) if g_an else np.nan,
                             delta_an=d_an, gamma_an=g_an))
    return pd.DataFrame(rows)


def plot_error_curves(df):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    colors = {'FORWARD': '#1f77b4', 'BACKWARD': '#ff7f0e', 'CENTRAL': '#2ca02c'}
    for j, fd in enumerate(['FORWARD', 'BACKWARD', 'CENTRAL']):
        for i, (col, title) in enumerate([('delta_err', 'Delta |abs error|'), ('gamma_err', 'Gamma relative error')]):
            ax = axes[i, j]
            sub = df[df.fd == fd]
            for bt, ls in [('RELATIVE', '-'), ('ABSOLUTE', '--')]:
                g = sub[sub.bump_type == bt].groupby('grid')[col].median()
                ax.loglog(g.index, g.values, ls, color=colors[fd], label=f'{bt.lower()} bump (median)')
            ax.set_title(f'{fd.title()} - {title}')
            ax.grid(True, which='both', alpha=.3)
            if i == 1:
                ax.set_xlabel('bump size h  (relative: fraction of S;  absolute: currency units)')
            if j == 0:
                ax.set_ylabel('error')
            ax.legend(fontsize=8)
    fig.suptitle('Numerical error of bump-and-revalue greeks (median over 108 European options)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_error_vs_bump.png'), dpi=130)
    plt.close(fig)


def plot_abs_vs_rel_spread(df):
    """Absolute bump: different stocks (spot 130 ... 250) have different optimum; relative bump: they line up."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    sub = df[(df.fd == 'CENTRAL') & (df.kind == 'CALL') & (df.moneyness == 1.0) & (df.expiry == 1.0)]
    for ax, bt in zip(axes, ['ABSOLUTE', 'RELATIVE']):
        for stock, g in sub[sub.bump_type == bt].groupby('stock'):
            ax.loglog(g['grid'], g['delta_err'], label=f"{stock} (S={g.spot.iloc[0]:.0f})")
        ax.set_title(f'Central delta error, {bt.lower()} bump, ATM 1y call')
        ax.set_xlabel('bump size ' + ('h (currency)' if bt == 'ABSOLUTE' else 'h / S'))
        ax.grid(True, which='both', alpha=.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('|delta error|')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_absolute_vs_relative.png'), dpi=130)
    plt.close(fig)


def plot_option_heatmap(df, best):
    """Error at the recommended relative bump for every option - shows where finite differences are worst."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (greek, col) in zip(axes, [('Delta', 'delta_err'), ('Gamma', 'gamma_err')]):
        h = best[('CENTRAL', greek)]
        sub = df[(df.fd == 'CENTRAL') & (df.bump_type == 'RELATIVE')]
        sub = sub[np.isclose(sub.grid, REL_GRID[np.argmin(abs(REL_GRID - h))])]
        piv = sub.groupby(['expiry', 'moneyness'])[col].median().unstack()
        im = ax.imshow(np.log10(piv.values), cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel('strike / spot')
        ax.set_ylabel('expiry')
        ax.set_title(f'{greek}: log10 error, central, relative h={h:.1e}')
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_error_by_option.png'), dpi=130)
    plt.close(fig)


def recommend(df):
    """Optimal bump = minimiser of the median error curve over all options (also report worst case)."""
    best = {}
    rows = []
    for fd, bt in itertools.product(['FORWARD', 'BACKWARD', 'CENTRAL'], ['RELATIVE', 'ABSOLUTE']):
        for greek, col in [('Delta', 'delta_err'), ('Gamma', 'gamma_err')]:
            sub = df[(df.fd == fd) & (df.bump_type == bt)]
            med = sub.groupby('grid')[col].median()
            worst = sub.groupby('grid')[col].max()
            h = med.idxmin()
            rows.append(dict(method=fd, bump=bt, greek=greek, best_bump=h, median_err=med.min(),
                             worst_err_at_that_bump=worst[h], best_worst_case_err=worst.min()))
            if bt == 'RELATIVE':
                best[(fd, greek)] = h
    return pd.DataFrame(rows), best


def per_stock_optimum(df):
    """Optimal central-delta bump for each stock separately: relative optimum is stable, absolute is not."""
    rows = []
    for bt in ['RELATIVE', 'ABSOLUTE']:
        for stock, g in df[(df.fd == 'CENTRAL') & (df.bump_type == bt)].groupby('stock'):
            h = g.groupby('grid')['delta_err'].median().idxmin()
            rows.append(dict(bump=bt, stock=stock, spot=g.spot.iloc[0], best_bump=h, best_bump_over_spot=h if bt == 'RELATIVE' else h / g.spot.iloc[0]))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = run_study()
    df.to_csv(os.path.join(OUT, 'study_results.csv'), index=False)
    table, best = recommend(df)
    table.to_csv(os.path.join(OUT, 'optimal_bumps.csv'), index=False)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False, float_format=lambda x: f'{x:.3e}'))
    ps = per_stock_optimum(df)
    ps.to_csv(os.path.join(OUT, 'optimal_bump_per_stock.csv'), index=False)
    print(ps.to_string(index=False, float_format=lambda x: f'{x:.3e}'))
    plot_error_curves(df)
    plot_abs_vs_rel_spread(df)
    plot_option_heatmap(df, best)
    print('done')

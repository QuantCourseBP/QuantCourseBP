# Homework 3, Q2: a digital call is the limit of (scaled) bull call spreads.
# Uses EuropeanAnalyticPricer (lecture) for the spreads and EuropeanDigitalAnalyticPricer (Q1) for the digital;
# the plots are drawn with my_plotter of Homework 1 (plotter.py).
# Run from this folder with the course venv:  python q2_bull_spread.py   (output: q2_bull_spread.out, figures/)
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
root = next((p for p in [here, *here.parents] if (p / 'src' / 'enums.py').exists()), here)
sys.path[:0] = [str(here), str(root)]               # this folder (library) and the repository (src.enums)

import matplotlib
matplotlib.use('Agg')                                # files only (no window)
import matplotlib.pyplot as plt
import numpy as np
from pricer import *                                 # library of Lecture 5 + the classes of Q1
from plotter import my_plotter                       # Homework 1 (HW 2/1)

stock, strike, expiry = Stock.BLUECHIP_BANK, 130.0, 1.0
model = MarketModel(stock)


def bull_spread(k_low: float, k_high: float) -> list[EuropeanAnalyticPricer]:
    """Bull call spread (Definition 2): LONG call with strike k_low + SHORT call with strike k_high."""
    long_leg = EuropeanContract(stock, PutCallFwd.CALL, LongShort.LONG, k_low, expiry)
    short_leg = EuropeanContract(stock, PutCallFwd.CALL, LongShort.SHORT, k_high, expiry)
    return [EuropeanAnalyticPricer(long_leg, model, Params()), EuropeanAnalyticPricer(short_leg, model, Params())]


def scaled_value(legs: list[EuropeanAnalyticPricer], h: float) -> float:
    """Price of (1/h) x bull spread: the sum of the two legs (the SHORT leg is negative), divided by h."""
    return sum(leg.calc_fair_value() for leg in legs) / h


def scaled_payoff(legs: list[EuropeanAnalyticPricer], h: float, s_T: float) -> float:
    return sum(leg.contract.payoff({expiry: s_T}) for leg in legs) / h


def at_spot(pricer_value, s: float) -> float:
    """Value with the model's spot set to s (the model is shared by all pricers), then restore it."""
    base = model.spot
    model.spot = s
    value = pricer_value()
    model.spot = base
    return value


digital_contract = EuropeanDigitalContract(stock, PutCallFwd.CALL, LongShort.LONG, strike, expiry)
digital = EuropeanDigitalAnalyticPricer(digital_contract, model, Params())
c_d = digital.calc_fair_value()
print(f'market: S = {model.spot}, vol = {model.vol}, r = {model.risk_free_rate};  K = {strike}, T = {expiry}')
print(f'digital call C_d = {c_d:.6f}')

# 1. table: right spread R_h = (1/h)(K, K+h) <= C_d <= left spread L_h = (1/h)(K-h, K)  (proof of Lemma 4)
print(f"{'h':>7s}{'R_h':>11s}{'L_h':>11s}{'C_d - R_h':>12s}{'L_h - C_d':>12s}")
widths = np.array([40.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01])
right = np.array([scaled_value(bull_spread(strike, strike + h), h) for h in widths])
left = np.array([scaled_value(bull_spread(strike - h, strike), h) for h in widths])
for h, r_h, l_h in zip(widths, right, left):
    print(f'{h:7.2f}{r_h:11.6f}{l_h:11.6f}{c_d - r_h:12.2e}{l_h - c_d:12.2e}')
slope = np.polyfit(np.log(widths[-4:]), np.log(c_d - right[-4:]), 1)[0]
print(f'error of R_h ~ h^{slope:.2f} for small h (first order)')

# 2. figure: (a) payoffs at T, (b) prices today against the spot, (c) price error against h
plot_widths = [20.0, 10.0, 2.0]
colors = ['tab:green', 'tab:orange', 'tab:blue', 'black']
names = [f'bull spread / h, h = {h:g}' for h in plot_widths] + ['digital call']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

s_T = np.linspace(100.0, 160.0, 601)
payoffs = [[scaled_payoff(bull_spread(strike, strike + h), h, s) for s in s_T] for h in plot_widths]
payoffs.append([digital.contract.payoff({expiry: s}) for s in s_T])
my_plotter(s_T, payoffs, names=names, ax=axes[0],
           layout={'title': '(a) payoff at expiry', 'x_label': '$S_T$', 'y_label': 'payoff',
                   'colors': colors, 'linestyles': ['-', '-', '-', '--'], 'linewidth': 2, 'title_fontsize': 14,
                   'label_fontsize': 12, 'legend_fontsize': 10, 'title_pad': 10, 'label_pad': 5,
                   'hline': None, 'vlines': [strike], 'grid': True, 'legend_loc': 'upper left'})

spots = np.linspace(60.0, 220.0, 161)
prices = [[at_spot(lambda: scaled_value(bull_spread(strike, strike + h), h), s) for s in spots] for h in plot_widths]
prices.append([at_spot(digital.calc_fair_value, s) for s in spots])
my_plotter(spots, prices, names=names, ax=axes[1],
           layout={'title': '(b) price today', 'x_label': 'spot $S$', 'y_label': 'price',
                   'colors': colors, 'linestyles': ['-', '-', '-', '--'], 'linewidth': 2, 'title_fontsize': 14,
                   'label_fontsize': 12, 'legend_fontsize': 10, 'title_pad': 10, 'label_pad': 5,
                   'hline': None, 'vlines': [strike], 'grid': True, 'legend_loc': 'upper left'})

my_plotter(widths, [c_d - right, left - c_d], names=['$C_d - R_h$ (right spread)', '$L_h - C_d$ (left spread)'],
           ax=axes[2],
           layout={'title': '(c) error against the width h', 'x_label': 'h (log scale)',
                   'y_label': 'error (log scale)', 'colors': ['tab:blue', 'tab:red'], 'markers': ['o', 's'],
                   'linewidth': 2, 'title_fontsize': 14, 'label_fontsize': 12, 'legend_fontsize': 10,
                   'title_pad': 10, 'label_pad': 5, 'hline': None, 'logx': True, 'logy': True, 'grid': True,
                   'legend_loc': 'upper left'})
fig.suptitle(f'Digital call = limit of bull call spreads  (BLUECHIP_BANK, K = {strike:g}, T = {expiry:g},'
             f' C_d = {c_d:.4f})', fontsize=15)
fig.tight_layout()
(here / 'figures').mkdir(exist_ok=True)
fig.savefig(here / 'figures' / 'q2_digital_vs_bull_spreads.pdf')
fig.savefig(here / 'figures' / 'q2_digital_vs_bull_spreads.png', dpi=110)
print('figure saved: figures/q2_digital_vs_bull_spreads.pdf (and .png)')

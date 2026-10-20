"""Part b): a digital call option is the limit of (scaled) bull spread call options.

Bull spread with width w: long call at K - w/2, short call at K + w/2, divided by w.
As w -> 0 its payoff and its price converge to those of the digital call with strike K.

Run from the repository root:
    python homeworks\\29481ec_library\\bull_spread_figure.py
"""
import sys
from pathlib import Path

# make `src.enums` importable no matter where the script is started from
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from pricer import *

UNDERLYING = Stock.BLUECHIP_BANK
EXPIRY = 1.0
WIDTHS = [30.0, 15.0, 5.0, 1.0]          # bull spread widths (strike distance)


def digital_call(strike):
    return EuropeanDigitalContract(UNDERLYING, PutCallFwd.CALL, LongShort.LONG, strike, EXPIRY)


def bull_spread_legs(strike, width):
    long_call = EuropeanContract(UNDERLYING, PutCallFwd.CALL, LongShort.LONG, strike - width / 2, EXPIRY)
    short_call = EuropeanContract(UNDERLYING, PutCallFwd.CALL, LongShort.SHORT, strike + width / 2, EXPIRY)
    return long_call, short_call


def bull_spread_payoff(strike, width, spot_value):
    t = round(EXPIRY, Contract.timeline_digits)
    legs = bull_spread_legs(strike, width)
    return sum(leg.payoff({t: spot_value}) for leg in legs) / width


def bull_spread_value(strike, width, spot_value):
    model = MarketModel(UNDERLYING)
    model.spot = spot_value
    legs = bull_spread_legs(strike, width)
    return sum(EuropeanAnalyticPricer(leg, model, Params()).calc_fair_value() for leg in legs) / width


def digital_value(strike, spot_value):
    model = MarketModel(UNDERLYING)
    model.spot = spot_value
    return EuropeanDigitalAnalyticPricer(digital_call(strike), model, Params()).calc_fair_value()


def main():
    strike = round(MarketData.get_spot()[UNDERLYING])         # K = 130
    spots = np.linspace(strike * 0.7, strike * 1.3, 601)
    t = round(EXPIRY, Contract.timeline_digits)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(WIDTHS)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    # 1) payoff at expiry
    ax = axes[0]
    for width, color in zip(WIDTHS, colors):
        ax.plot(spots, [bull_spread_payoff(strike, width, s) for s in spots],
                color=color, label=f'bull spread / w, w={width:g}')
    ax.plot(spots, [digital_call(strike).payoff({t: s}) for s in spots],
            'r--', linewidth=2, label='digital call')
    ax.set_title('Payoff at expiry')
    ax.set_xlabel('Spot at expiry')
    ax.set_ylabel('Payoff')

    # 2) fair value today as function of spot
    ax = axes[1]
    for width, color in zip(WIDTHS, colors):
        ax.plot(spots, [bull_spread_value(strike, width, s) for s in spots],
                color=color, label=f'bull spread / w, w={width:g}')
    ax.plot(spots, [digital_value(strike, s) for s in spots],
            'r--', linewidth=2, label='digital call')
    ax.set_title('Fair value today')
    ax.set_xlabel('Spot today')
    ax.set_ylabel('Fair value')

    # 3) convergence of the price at the current spot
    ax = axes[2]
    spot0 = MarketData.get_spot()[UNDERLYING]
    widths = np.logspace(-2, 1.5, 30)
    target = digital_value(strike, spot0)
    errors = [abs(bull_spread_value(strike, w, spot0) - target) for w in widths]
    ax.loglog(widths, errors, 'o-')
    ax.set_title('Price difference at current spot')
    ax.set_xlabel('Bull spread width w')
    ax.set_ylabel('|bull spread / w  -  digital call|')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'A digital call (K={strike}, T={EXPIRY}) as the limit of bull spreads '
                 f'(width w -> 0)', fontsize=13)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / 'digital_vs_bull_spread.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('Figure saved to', out)


if __name__ == '__main__':
    main()

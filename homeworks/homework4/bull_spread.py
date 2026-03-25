import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from pricer import *

UNDERLYING = Stock.BLUECHIP_BANK
EXPIRY = 1.0
model = MarketModel(UNDERLYING)
K = model.spot  # ATM strike
params = Params()

spots = np.linspace(0.5 * K, 1.5 * K, 400)
eps_values = [20.0, 10.0, 5.0, 1.0, 0.1]


def vanilla_call_fv(spot_val, strike):
    m = MarketModel(UNDERLYING)
    m.spot = spot_val
    c = EuropeanContract(UNDERLYING, PutCallFwd.CALL, LongShort.LONG, strike, EXPIRY)
    return EuropeanAnalyticPricer(c, m, params).calc_fair_value()


def digital_call_fv(spot_val):
    m = MarketModel(UNDERLYING)
    m.spot = spot_val
    c = EuropeanDigitalContract(UNDERLYING, PutCallFwd.CALL, LongShort.LONG, K, EXPIRY)
    return EuropeanDigitalAnalyticPricer(c, m, params).calc_fair_value()


# compute curves
digital_curve = np.array([digital_call_fv(s) for s in spots])

bull_spread_curves = {}
for eps in eps_values:
    curve = np.array(
        [(vanilla_call_fv(s, K - eps) - vanilla_call_fv(s, K + eps)) / (2 * eps)
         for s in spots]
    )
    bull_spread_curves[eps] = curve

# plot
plt.figure(figsize=(10, 6))
colors = plt.cm.plasma(np.linspace(0.15, 0.80, len(eps_values)))

for (eps, curve), color in zip(bull_spread_curves.items(), colors):
    plt.plot(spots, curve, color=color, lw=1.5, label=f'Bull spread  ε={eps}')

plt.plot(spots, digital_curve, 'k--', lw=2.5, label='Digital call (exact)')
plt.axvline(K, color='grey', lw=0.8, linestyle=':', label=f'Strike K={K:.1f}')

plt.xlabel('Spot price S₀')
plt.ylabel('Price')
plt.title(f'Digital Call as Limit of Bull Spread  [{UNDERLYING.name}, T={EXPIRY}y]')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('digital_vs_bull_spread.png', dpi=150)
plt.show()
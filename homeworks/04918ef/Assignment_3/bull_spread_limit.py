import matplotlib.pyplot as plt
import numpy as np
from contract import EuropeanContract, EuropeanDigitalContract
from model import MarketModel
from pricer import EuropeanAnalyticPricer, EuropeanDigitalAnalyticPricer, Params
from src.enums import *

# 1. Setup market model and base parameters
underlying = Stock.BLUECHIP_BANK
model = MarketModel(underlying)
params = Params()

K1 = 130.0  # Base strike price
expiry = 1.0  # Time to maturity in years

# 2. Price European Digital Call option benchmark
digital_contract = EuropeanDigitalContract(
    underlying, PutCallFwd.CALL, LongShort.LONG, K1, expiry
)
digital_pricer = EuropeanDigitalAnalyticPricer(
    digital_contract, model, params
)
digital_price = digital_pricer.calc_fair_value()

# 3. Price Bull Call Spreads across shrinking dK (delta K) steps
dK_list = [20.0, 10.0, 5.0, 1.0, 0.1, 0.01]
bull_spread_prices = []

for dK in dK_list:
    K2 = K1 + dK
    # Long Call at K1
    call1 = EuropeanContract(
        underlying, PutCallFwd.CALL, LongShort.LONG, K1, expiry
    )
    pricer1 = EuropeanAnalyticPricer(call1, model, params)

    # Short Call at K2
    call2 = EuropeanContract(
        underlying, PutCallFwd.CALL, LongShort.LONG, K2, expiry
    )
    pricer2 = EuropeanAnalyticPricer(call2, model, params)

    # Scaled Bull Spread Fair Value: (C(K1) - C(K2)) / dK
    bull_spread_price = (
        pricer1.calc_fair_value() - pricer2.calc_fair_value()
    ) / dK
    bull_spread_prices.append(bull_spread_price)

# 4. Plot results
plt.figure(figsize=(10, 6))
plt.plot(
    dK_list,
    bull_spread_prices,
    "o-",
    label=r"Scaled Bull Spread Price: $(C(K_1) - C(K_1 + \Delta K)) / \Delta K$",
)
plt.axhline(
    y=digital_price,
    color="r",
    linestyle="--",
    label=f"Digital Call Fair Value ({digital_price:.6f})",
)

plt.xscale("log")
plt.gca().invert_xaxis()  # Reversing x-axis to show dK approaching 0 from left to right
plt.xlabel(r"$\Delta K$ (log scale, approaching 0)", fontsize=12)
plt.ylabel("Fair Value", fontsize=12)
plt.title(
    "Digital Call Option as the Limit of Scaled Bull Spread Call Options",
    fontsize=14,
)
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
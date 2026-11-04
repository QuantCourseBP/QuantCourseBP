
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pricer import (EuropeanAnalyticPricer, FiniteDifferenceMethod, BumpType, Params)

from contract import EuropeanContract, PutCallFwd, LongShort
from model import MarketModel
from market_data import MarketData


#create set of European call options

spots = [20, 100, 500]
strike_ratios = [0.8, 1.0, 1.2]
expiries = [0.1, 1.0, 2.0]
volatilities = [0.15, 0.30]

option_types = [PutCallFwd.CALL, PutCallFwd.PUT]

rate = 0.03

bump_sizes = np.logspace(-7, -1, 25)

underlying = next(iter(MarketData.get_spot()))

results = []

# analytical and numerical greeks comparison


for spot in spots:
    for strike_ratio in strike_ratios:
        for expiry in expiries:
            for vol in volatilities:
                for option_type in option_types:

                    strike = spot * strike_ratio

                    contract = EuropeanContract(
                        underlying=underlying,
                        derivative_type=option_type,
                        long_short=LongShort.LONG,
                        strike=strike,
                        expiry=expiry
                    )

                    model = MarketModel(underlying)

                    model.spot = spot
                    model.vol = vol
                    model.risk_free_rate = rate

                    pricer = EuropeanAnalyticPricer(
                        contract=contract,
                        model=model,
                        params=Params()
                    )

                    # Analytical Greeks
                    delta_true = pricer.calc_delta()
                    gamma_true = pricer.calc_gamma()

                    # All six configurations
                    for method in FiniteDifferenceMethod:
                        for bump_type in BumpType:

                            pricer.finite_difference_method = method
                            pricer.bump_type = bump_type

                            for bump in bump_sizes:

                                delta_fd = pricer.calc_delta_fd(
                                    bump_size=bump
                                )

                                gamma_fd = pricer.calc_gamma_fd(
                                    bump_size=bump
                                )

                                delta_error = abs(
                                    delta_fd - delta_true
                                )

                                gamma_error = abs(
                                    gamma_fd - gamma_true
                                )

                                results.append({
                                    "spot": spot,
                                    "strike": strike,
                                    "expiry": expiry,
                                    "vol": vol,
                                    "option_type": option_type.name,
                                    "method": method.value,
                                    "bump_type": bump_type.value,
                                    "bump_size": bump,
                                    "delta_error": delta_error,
                                    "gamma_error": gamma_error,
                                    "scaled_gamma_error": spot * gamma_error
                                })


df = pd.DataFrame(results)


#check errors

summary = (
    df.groupby(
        ["method", "bump_type", "bump_size"]
    )
    .agg(delta_error=("delta_error", "mean"),gamma_error=("gamma_error", "mean"))

    .reset_index()
)

#graphs

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 9)
)

for i, greek in enumerate(["delta", "gamma"]):
    for j, bump_type in enumerate(BumpType):

        ax = axes[i, j]

        subset_type = summary[
            summary["bump_type"] == bump_type.value
        ]

        for method in FiniteDifferenceMethod:

            subset = subset_type[
                subset_type["method"] == method.value
            ]

            ax.loglog(
                subset["bump_size"],
                subset[f"{greek}_error"],
                marker="o",
                markersize=3,
                label=method.value
            )

        ax.set_title(
            f"{greek.capitalize()} - {bump_type.value} bump"
        )

        ax.set_xlabel("Bump size")

        if greek == "delta":
            ax.set_ylabel("Mean absolute error (Delta error)")
        else:
            ax.set_ylabel("Mean absolute error (Spot x Gamma error)")

        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

fig.suptitle(
    "Finite Difference Greeks: Numerical Error",
    fontsize=15
)

plt.tight_layout()
plt.show()


#check which bump sizes are optimal

print("\nbump sizes")

for greek in ["delta", "gamma"]:
    for method in FiniteDifferenceMethod:
        for bump_type in BumpType:

            subset = summary[
                (summary["method"] == method.value)
                & (summary["bump_type"] == bump_type.value)
            ]

            best = subset.loc[
                subset[f"{greek}_error"].idxmin()
            ]

            print(
                f"{greek.upper():5s} | "
                f"{method.value:8s} | "
                f"{bump_type.value:8s} | "
                f"bump = {best['bump_size']:.2e} | "
                f"error = {best[f'{greek}_error']:.2e}"
            )

print("\nex: european call")

contract = EuropeanContract(
    underlying=underlying,
    derivative_type=PutCallFwd.CALL,
    long_short=LongShort.LONG,
    strike=100.0,
    expiry=1.0
)

model = MarketModel(underlying)

model.spot = 100.0
model.vol = 0.2
model.risk_free_rate = 0.05

pricer = EuropeanAnalyticPricer(
    contract, model, Params()
)

pricer.finite_difference_method = FiniteDifferenceMethod.CENTRAL
pricer.bump_type = BumpType.RELATIVE

print("Analytical Delta:", pricer.calc_delta())
print("Numerical Delta: ", pricer.calc_delta_fd(1e-5))

print("Analytical Gamma:", pricer.calc_gamma())
print("Numerical Gamma: ", pricer.calc_gamma_fd(1e-4))

#output with comments

# C:\Users\tolya\PycharmProjects\QuantCourseBP\.venv\Scripts\python.exe C:\Users\tolya\PycharmProjects\QuantCourseBP\homeworks\51029fc\hw4\solution\greeks_test.py
# QuantCourseBP b6d636fd2043e22bf6f249c9ee0af1a17290abcb*
#
# OPTIMAL BUMP SIZES
# =================================================================
# DELTA | forward  | absolute | bump = 5.62e-07 | error = 2.79e-08
# DELTA | forward  | relative | bump = 1.00e-07 | error = 7.87e-08
# DELTA | backward | absolute | bump = 1.00e-06 | error = 2.65e-08
# DELTA | backward | relative | bump = 1.00e-07 | error = 7.89e-08
# DELTA | central  | absolute | bump = 1.78e-04 | error = 5.66e-11
# DELTA | central  | relative | bump = 3.16e-06 | error = 1.23e-11  SMALLEST ERROR => optimal
# GAMMA | forward  | absolute | bump = 1.78e-04 | error = 1.43e-06
# GAMMA | forward  | relative | bump = 3.16e-06 | error = 4.96e-07
# GAMMA | backward | absolute | bump = 1.78e-04 | error = 1.46e-06
# GAMMA | backward | relative | bump = 3.16e-06 | error = 4.56e-07
# GAMMA | central  | absolute | bump = 1.78e-03 | error = 9.87e-09
# GAMMA | central  | relative | bump = 5.62e-05 | error = 1.94e-09  SMALLEST ERROR => optimal
#
#   central + relative is optimal
# ex: european call
# =================================================================
# Analytical Delta: 0.6368306511756191
# Numerical Delta:  0.636830651092879
# Analytical Gamma: 0.018762017345846895
# Numerical Gamma:  0.018762017077733617
#
# Process finished with exit code 0

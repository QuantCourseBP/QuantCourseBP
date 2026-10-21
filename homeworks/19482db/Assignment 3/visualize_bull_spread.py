import numpy as np
import matplotlib.pyplot as plt

from contract import EuropeanContract, EuropeanDigitalContract
from model import MarketModel
from pricer import (
    Params,
    EuropeanAnalyticPricer,
    EuropeanDigitalAnalyticPricer,
)
from src.enums import Stock, PutCallFwd, LongShort


def plot_digital_call_limit():
    # Market and contract parameters
    underlying = Stock.BLUECHIP_BANK
    model = MarketModel(underlying)

    spot = model.spot
    strike = spot
    expiry = 1.0

    # Stock prices at expiry
    stock_prices = np.linspace(0.75 * spot, 1.25 * spot, 1500)

    # Digital call payoff
    digital_contract = EuropeanDigitalContract(
        underlying,
        PutCallFwd.CALL,
        LongShort.LONG,
        strike,
        expiry,
    )

    digital_payoffs = np.array([
        digital_contract.payoff({round(expiry, 6): s})
        for s in stock_prices
    ])

    # Plot the digital call payoff
    plt.figure(figsize=(10, 6))
    plt.step(
        stock_prices,
        digital_payoffs,
        where="post",
        color="black",
        linewidth=2.5,
        label="Digital call",
    )

    # Bull spreads with progressively smaller strike differences
    strike_gaps = [
        0.10 * spot,
        0.05 * spot,
        0.02 * spot,
        0.01 * spot,
        0.005 * spot,
    ]

    for h in strike_gaps:
        lower_call = EuropeanContract(
            underlying,
            PutCallFwd.CALL,
            LongShort.LONG,
            strike,
            expiry,
        )

        upper_call = EuropeanContract(
            underlying,
            PutCallFwd.CALL,
            LongShort.SHORT,
            strike + h,
            expiry,
        )

        bull_spread_payoffs = np.array([
            (
                lower_call.payoff({round(expiry, 6): s})
                + upper_call.payoff({round(expiry, 6): s})
            ) / h
            for s in stock_prices
        ])

        plt.plot(
            stock_prices,
            bull_spread_payoffs,
            linewidth=1.5,
            label=f"Bull spread, h = {h:.3f}",
        )

    # Mark the strike price
    plt.axvline(
        strike,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"Strike K = {strike:.2f}",
    )

    # Configure the figure
    plt.xlabel("Stock price at expiry ($)")
    plt.ylabel("Payoff / h")
    plt.title("A Digital Call as the Limit of Bull Spreads")
    plt.ylim(-0.05, 1.15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save and display the figure
    plt.savefig("digital_call_bull_spread.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    plot_digital_call_limit()
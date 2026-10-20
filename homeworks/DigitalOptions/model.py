from __future__ import annotations
from market_data import *
from src.enums import *
import numpy as np


# TASK:
# 1. Define MarketModel class.
# 2. Define its __init__() method:
#    a. Take an underlying Stock as input.
#    b. Store the underlying and the corresponding market data (risk-free rate, spot, volatility).
# 3. Implement calc_df() method which takes a tenor and returns the corresponding discount factor.


class MarketModel:
    def __init__(self, underlying: Stock) -> None:
        self.underlying: Stock = underlying
        self.risk_free_rate: float = MarketData.get_risk_free_rate()
        self.spot: float = MarketData.get_spot()[self.underlying]
        self.vol: float = MarketData.get_vol()[self.underlying]

    def calc_df(self, tenor: float) -> float:
        return np.exp(-1.0 * self.risk_free_rate * tenor)

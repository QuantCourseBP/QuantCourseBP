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
    def __init__(self, underlying: Stock):
        self.underlying = underlying
        self.risk_free_rate = MarketData.get_risk_free_rate()
        self.spot = MarketData.get_spot()[self.underlying]
        self.volatility = MarketData.get_vol()[self.underlying]

    def calc_df(self, tenor: float):
        return np.exp(-self.risk_free_rate * tenor)

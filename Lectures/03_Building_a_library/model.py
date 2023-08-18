from __future__ import annotations
from market_data import *
from src.enums import *
import numpy as np


# TASK:
# Implement __init__() method for MarketModel class. Take an underlying as input.
# Store market data (risk-free rate, spot, volatility) of the given underlying.
# Implement get_df() method which takes a tenor and returns the corresponding discount factor.


class MarketModel:
    def get_rate(self) -> float:
        return self._interest_rate

    def get_spot(self) -> float:
        return self._spot

    def get_vol(self) -> float:
        return self._volatility

    def get_df(self, tenor: float) -> float:
        pass

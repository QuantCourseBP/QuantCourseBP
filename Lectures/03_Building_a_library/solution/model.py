from __future__ import annotations
from market_data import *
from src.enums import *
import numpy as np


class MarketModel:
    def __init__(self, und: Stock) -> None:
        self._interest_rate: float = MarketData.get_risk_free_rate()
        self._spot: float = MarketData.get_spot()[und]
        self._volatility: float = MarketData.get_vol()[und]

    def get_rate(self) -> float:
        return self._interest_rate

    def get_spot(self) -> float:
        return self._spot

    def get_vol(self) -> float:
        return self._volatility

    def get_df(self, tenor: float) -> float:
        return np.exp(-1.0 * self._interest_rate * tenor)

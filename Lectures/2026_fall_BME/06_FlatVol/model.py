from __future__ import annotations
from abc import ABC, abstractmethod
from market_data import *
from enums import *
import numpy as np


# TASK:
# Implement BSVol and FlatVol models in the pricing library


class MarketModel:
    def __init__(self, underlying: Stock) -> None:
        self.underlying: Stock = underlying
        self.risk_free_rate: float = MarketData.get_risk_free_rate()
        self.spot: float = MarketData.get_spot()[self.underlying]
        self.vol: float = MarketData.get_vol()[self.underlying]

    def calc_df(self, tenor: float) -> float:
        return np.exp(-1.0 * self.risk_free_rate * tenor)

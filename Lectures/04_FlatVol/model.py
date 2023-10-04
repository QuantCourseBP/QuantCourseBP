from __future__ import annotations
from abc import ABC, abstractmethod
from market_data import *
from enums import *
import numpy as np


# TASK:
# Implement BSVol and FlatVol models in the pricing library


class MarketModel(ABC):
    def __init__(self, underlying: Stock) -> None:
        self.underlying: Stock = underlying
        self.risk_free_rate: float = MarketData.get_risk_free_rate()
        self.spot: float = MarketData.get_spot()[self.underlying]
        self.volgrid: VolGrid = MarketData.get_volgrid()[self.underlying]

    def bump_rate(self, bump_size: float) -> None:
        self.risk_free_rate += bump_size

    def bump_spot(self, bump_size: float) -> None:
        self.spot += bump_size

    def bump_volgrid(self, bump_size: float) -> None:
        self.volgrid.values += bump_size

    def calc_df(self, tenor: float) -> float:
        return np.exp(-1.0 * self.risk_free_rate * tenor)

    @abstractmethod
    def get_vol(self, strike: float, expiry: float) -> float:
        pass

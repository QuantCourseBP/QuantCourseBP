from __future__ import annotations
from market_data import MarketData
from enums import *
from market_data import VolGrid
import numpy as np
from abc import ABC, abstractmethod
from src.enums import *

class MarketModel(ABC):
    def __init__(self, und: Stock):
        self._und = und
        self._interest_rate = MarketData.get_risk_free_rate()
        self._initial_spot = MarketData.get_initial_spot()[self._und]
        self._volgrid = MarketData.get_vol_grid()[self._und]

    def get_rate(self) -> float:
        return self._interest_rate

    def get_initial_spot(self) -> float:
        return self._initial_spot

    def get_volgrid(self) -> VolGrid:
        return self._volgrid

    def get_df(self, t: float) -> float:
        return np.exp(-self.interest_rate * t)

    @abstractmethod
    def get_simulated_spot(self, t: float, Z: float) -> float:
        pass

    @staticmethod
    def get_models() -> dict[str, MarketModel]:
        return {cls.__name__: cls for cls in MarketModel.__subclasses__()}


class BSVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)

    def get_vol(self) -> float:
        return self.get_volgrid.get_vol(1.0, 1.0)

    def get_simulated_spot(self, t: float, Z: float) -> float:
        return np.exp(
            (self.interest_rate * t - 0.5 * self.get_vol() ** 2 * t) + (self.get_vol() * Z * np.sqrt(t)))


class FlatVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)

    def get_vol(self, t: float, strike: float) -> float:
        return self.get_volgrid.get_vol(t, strike)

    def get_simulated_spot(self, t: float, strike: float, Z: float) -> float:
        vol = self.get_vol(t, strike)
        return np.exp(
            (self.interest_rate * t - 0.5 * vol ** 2 * t) + (vol * Z * np.sqrt(t)))



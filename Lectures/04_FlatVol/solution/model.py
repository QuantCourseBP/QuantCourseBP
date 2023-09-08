from __future__ import annotations
from abc import ABC, abstractmethod
from src.market_data import *
from src.enums import *
import numpy as np


class MarketModel(ABC):
    def __init__(self, und: Stock) -> None:
        self._und: Stock = und
        self._interest_rate: float = MarketData.get_risk_free_rate()
        self._spot: float = MarketData.get_spot()[und]
        self._volgrid: VolGrid = MarketData.get_volgrid()[und]

    def get_rate(self) -> float:
        return self._interest_rate

    def bump_rate(self, bump_size: float) -> None:
        self._interest_rate += bump_size

    def get_spot(self) -> float:
        return self._spot

    def bump_spot(self, bump_size: float) -> None:
        self._spot += bump_size

    def bump_volgrid(self, bump_size: float) -> None:
        self._volgrid.values += bump_size

    @abstractmethod
    def get_vol(self, strike: float, expiry: float) -> float:
        pass

    def get_df(self, tenor: float) -> float:
        return np.exp(-1.0 * self._interest_rate * tenor)


class BSVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)
        # Reference spot is used to calculate ATM strike to imply the volatility.
        # Its value is not bumped in case of greek calculation.
        self.__reference_spot = MarketData.get_spot()[self._und]

    def get_vol(self, strike: float, expiry: float) -> float:
        atm_strike = 1.0 * self.__reference_spot
        expiry = 1.0
        coordinate = np.array([(atm_strike, expiry)])
        return self._volgrid.get_vol(coordinate)[0]


class FlatVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)

    def get_vol(self, strike: float, expiry: float) -> float:
        coordinate = np.array([(strike, expiry)])
        return self._volgrid.get_vol(coordinate)[0]

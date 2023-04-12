from __future__ import annotations
import math
from market_data import MarketData
from enums import *
from volgrid import VolGrid


class MarketModel:
    def __init__(self, und: Stock):
        self._und = und
        self.interest_rate = self.get_rate()
        self.spot = MarketData.get_initial_spot(self._und)

    def get_rate(self) -> float:
        pass

    @staticmethod
    def get_initial_spot() -> float:
        return MarketData.get_initial_spot()

    @staticmethod
    def get_volgrid() -> VolGrid:
        return MarketData.get_volgrid()

    def get_vol(self, exp: float, strike: float) -> float:
        return self.get_volgrid.get_vol(exp, strike)

    def get_df(self, exp: float) -> float:
        return math.exp(-self.interest_rate * exp)

    def get_underlying_level_at_exp(self, exp: float, Z: float) -> float:
        # Calculate stock price at expiry
        return self.spot * math.exp(
            (self.interest_rate * exp - 0.5 * self.sigma ** 2 * exp) + (self.vol * Z * math.sqrt(exp)))


class BSVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)
        self.vol = self.get_vol()

    def get_vol(self, exp=1.0, strike=1.0) -> float:
        return self.get_volgrid.get_vol(exp, strike)


class FlatVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)
        self.vol = self.get_vol()

    def get_vol(self, exp: float, strike: float) -> float:
        return self.get_volgrid.get_vol(exp, strike)

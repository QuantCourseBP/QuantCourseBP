from __future__ import annotations
from abc import ABC, abstractmethod
from market_data import *
from enums import *
import numpy as np


# TASKS:
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


class BSVolModel(MarketModel):
    def __init__(self, underlying: Stock):
        super().__init__(underlying)
        # Reference spot is used to calculate ATM strike to imply the volatility.
        # Its value is not bumped in case of greek calculation.
        self.reference_spot = MarketData.get_spot()[self.underlying]

    def get_vol(self, strike: float, expiry: float) -> float:
        """
        In Black-Scholes volatility model, we assume the volatility surface is flat at level of
        (strike=ATM, expiry=1.0).
        :param strike: Strike of option contract. Ignored.
        :param expiry: Expiry of option contract. Ignored.
        :return: Implied volatility.
        """
        atm_strike = 1.0 * self.reference_spot
        expiry = 1.0
        coordinate = np.array([(atm_strike, expiry)])
        return self.volgrid.get_vol(coordinate)[0]


class FlatVolModel(MarketModel):
    def __init__(self, underlying: Stock):
        super().__init__(underlying)

    def get_vol(self, strike: float, expiry: float) -> float:
        """
        In flat volatility model, we assume the volatility is flat for a given contract, defined by strike and expiry.
        :param strike: Strike of option contract.
        :param expiry: Expiry of option contract.
        :return: Implied volatility.
        """
        coordinate = np.array([(strike, expiry)])
        return self.volgrid.get_vol(coordinate)[0]

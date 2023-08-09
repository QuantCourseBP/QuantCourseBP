from __future__ import annotations
from src.market_data import MarketData
from src.enums import *
from src.market_data import VolGrid
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

    def bump_rate(self, bump_size: float) -> None:
        self._interest_rate += bump_size

    def get_initial_spot(self) -> float:
        return self._initial_spot

    def bump_initial_spot(self, bump_size: float) -> None:
        self._initial_spot += bump_size

    def bump_volgrid(self, bump_size: float) -> None:
        values = self._volgrid.get_values()
        values += bump_size
        self._volgrid = VolGrid(self._volgrid.get_und(), self._volgrid.get_points(), values)

    def get_df(self, tenor: float) -> float:
        return np.exp(-1.0 * self._interest_rate * tenor)

    @abstractmethod
    def get_vol(self, strike: float, expiry: float) -> float:
        pass

    @staticmethod
    def get_models() -> dict[str, MarketModel]:
        return {cls.__name__: cls for cls in MarketModel.__subclasses__()}


class BSVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)
        # Reference spot is used to calculate ATM strike to imply the volatility.
        # Its value is not bumped in case of greek calculation.
        self.__reference_spot = MarketData.get_initial_spot()[self._und]

    def get_vol(self, strike: float, expiry: float) -> float:
        """
        In Black-Scholes volatility model, we assume the volatility surface is flat at level of
        (strike=ATM, expiry=1.0).
        :param strike: Strike of option contract. Ignored.
        :param expiry: Expiry of option contract. Ignored.
        :return: Implied volatility.
        """
        atm_strike = 1.0 * self.__reference_spot
        expiry = 1.0
        coordinate = np.array([(atm_strike, expiry)])
        return self._volgrid.get_vol(coordinate)[0]

    def evolve_simulated_spot(self, vol: float, t_from: float, t_to: float, spot_from: float, z: float) -> float:
        rate = self._interest_rate
        dt = t_to - t_from
        return spot_from * np.exp((rate - 0.5 * vol**2) * dt + (vol * z * np.sqrt(dt)))


class FlatVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)

    def get_vol(self, strike: float, expiry: float) -> float:
        """
        In flat volatility model, we assume the volatility is flat for a given contract, defined by strike and expiry.
        :param strike: Strike of option contract.
        :param expiry: Expiry of option contract.
        :return: Implied volatility.
        """
        coordinate = np.array([(strike, expiry)])
        return self._volgrid.get_vol(coordinate)[0]

    def evolve_simulated_spot(self, vol: float, t_from: float, t_to: float, spot_from: float, z: float) -> float:
        rate = self._interest_rate
        dt = t_to - t_from
        return spot_from * np.exp((rate - 0.5 * vol**2) * dt + (vol * z * np.sqrt(dt)))

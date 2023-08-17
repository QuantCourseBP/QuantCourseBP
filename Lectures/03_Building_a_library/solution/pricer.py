from __future__ import annotations
from abc import ABC, abstractmethod
from scipy.stats import norm
from src.enums import *
from contract import *
from model import *


class Params:
    pass


class Pricer(ABC):
    __valuation_time: float = 0.0

    def __init__(self, contract: Contract, model: MarketModel, params: Params):
        self._contract: Contract = contract
        self._model: MarketModel = model
        self._params: Params = params

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    @abstractmethod
    def calc_delta(self) -> float:
        pass

    @abstractmethod
    def calc_gamma(self) -> float:
        pass

    @abstractmethod
    def calc_vega(self) -> float:
        pass

    @abstractmethod
    def calc_theta(self) -> float:
        pass

    @abstractmethod
    def calc_rho(self) -> float:
        pass

    def _get_valuation_time(self) -> float:
        return self.__valuation_time


class ForwardAnalyticPricer(Pricer):
    __supported_deriv_type: tuple[PutCallFwd, ...] = (PutCallFwd.FWD,)

    def __init__(self, contract: ForwardContract, model: MarketModel, params: Params):
        if not isinstance(contract, ForwardContract):
            raise TypeError(f'Contract must be of type ForwardContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self._contract.get_direction()
        spot = self._model.get_spot()
        strike = self._contract.get_strike()
        time_to_expiry = self._contract.get_expiry() - self._get_valuation_time()
        df = self._model.get_df(time_to_expiry)
        if self._contract.get_type() == PutCallFwd.FWD:
            return direction * (spot - strike * df)
        else:
            self._contract.raise_incorrect_derivative_type_error(self.__supported_deriv_type)

    def calc_delta(self) -> float:
        return self._contract.get_direction() * 1.0

    def calc_gamma(self) -> float:
        return 0.0

    def calc_vega(self) -> float:
        return 0.0

    def calc_theta(self) -> float:
        greek = 0.0
        strike = self._contract.get_strike()
        time_to_expiry = self._contract.get_expiry() - self._get_valuation_time()
        rate = self._model.get_rate()
        df = self._model.get_df(time_to_expiry)
        if self._contract.get_type() == PutCallFwd.FWD:
            greek = -1.0 * strike * df * rate
        else:
            self._contract.raise_incorrect_derivative_type_error(self.__supported_deriv_type)
        return self._contract.get_direction() * greek

    def calc_rho(self) -> float:
        greek = 0.0
        strike = self._contract.get_strike()
        time_to_expiry = self._contract.get_expiry() - self._get_valuation_time()
        df = self._model.get_df(time_to_expiry)
        if self._contract.get_type() == PutCallFwd.FWD:
            greek = strike * df * time_to_expiry
        else:
            self._contract.raise_incorrect_derivative_type_error(self.__supported_deriv_type)
        return self._contract.get_direction() * greek


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def d1(spot: float, strike: float, vol: float, rate: float, time_to_expiry: float):
        return 1 / (vol * np.sqrt(time_to_expiry)) * (np.log(spot / strike) + (rate + vol**2 / 2) * time_to_expiry)

    @staticmethod
    def d2(spot: float, strike: float, vol: float, rate: float, time_to_expiry: float):
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
        return d1 - vol * np.sqrt(time_to_expiry)

    def __init__(self, contract: EuropeanContract, model: MarketModel, params: Params):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self._contract.get_direction()
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        time_to_expiry = expiry - self._get_valuation_time()
        spot = self._model.get_spot()
        vol = self._model.get_vol()
        rate = self._model.get_rate()
        df = self._model.get_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
        if self._contract.get_type() == PutCallFwd.CALL:
            return direction * (spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
        elif self._contract.get_type() == PutCallFwd.PUT:
            return direction * (strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
        else:
            self._contract.raise_incorrect_derivative_type_error()

    def calc_delta(self ) -> float:
        greek = 0.0
        spot = self._model.get_spot()
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        time_to_expiry = expiry - self._get_valuation_time()
        vol = self._model.get_vol()
        rate = self._model.get_rate()
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
        if self._contract.get_type() == PutCallFwd.CALL:
            greek = norm.cdf(d1)
        elif self._contract.get_type() == PutCallFwd.PUT:
            greek = -norm.cdf(-d1)
        else:
            self._contract.raise_incorrect_derivative_type_error()
        return self._contract.get_direction() * greek

    def calc_gamma(self) -> float:
        greek = 0.0
        spot = self._model.get_spot()
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        time_to_expiry = expiry - self._get_valuation_time()
        vol = self._model.get_vol()
        rate = self._model.get_rate()
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
        if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
            greek = norm.pdf(d1) / (spot * vol * np.sqrt(time_to_expiry))
        else:
            self._contract.raise_incorrect_derivative_type_error()
        return self._contract.get_direction() * greek

    def calc_vega(self) -> float:
        greek = 0.0
        spot = self._model.get_spot()
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        time_to_expiry = expiry - self._get_valuation_time()
        vol = self._model.get_vol()
        rate = self._model.get_rate()
        df = self._model.get_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
        if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
            greek = strike * df * norm.pdf(d2) * np.sqrt(time_to_expiry)
        else:
            self._contract.raise_incorrect_derivative_type_error()
        return self._contract.get_direction() * greek

    def calc_theta(self) -> float:
        greek = 0.0
        strike = self._contract.get_strike()
        spot = self._model.get_spot()
        expiry = self._contract.get_expiry()
        time_to_expiry = expiry - self._get_valuation_time()
        vol = self._model.get_vol()
        rate = self._model.get_rate()
        df = self._model.get_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
        if self._contract.get_type() == PutCallFwd.CALL:
            greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                            + rate * strike * df * norm.cdf(d2))
        elif self._contract.get_type() == PutCallFwd.PUT:
            greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                            - rate * strike * df * norm.cdf(-d2))
        else:
            self._contract.raise_incorrect_derivative_type_error()
        return self._contract.get_direction() * greek

    def calc_rho(self) -> float:
        greek = 0.0
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        time_to_expiry = expiry - self._get_valuation_time()
        spot = self._model.get_spot()
        vol = self._model.get_vol()
        rate = self._model.get_rate()
        df = self._model.get_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
        if self._contract.get_type() == PutCallFwd.CALL:
            greek = strike * time_to_expiry * df * norm.cdf(d2)
        elif self._contract.get_type() == PutCallFwd.PUT:
            greek = -strike * time_to_expiry * df * norm.cdf(-d2)
        else:
            self._contract.raise_incorrect_derivative_type_error()
        return self._contract.get_direction() * greek

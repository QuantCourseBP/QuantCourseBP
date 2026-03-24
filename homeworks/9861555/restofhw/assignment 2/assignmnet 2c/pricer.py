from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm

from enums import *
from contract import *
from model import *


class Params:
    pass


class Pricer(ABC):
    valuation_time: float = 0.0

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params = params

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


class ForwardAnalyticPricer(Pricer):
    supported_deriv_type: tuple[PutCallFwd, ...] = (PutCallFwd.FWD,)

    def __init__(self, contract: ForwardContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, ForwardContract):
            raise TypeError(
                f'Contract must be of type ForwardContract but received {type(contract).__name__}'
            )
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        spot = self.model.spot
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry - self.valuation_time
        df = self.model.calc_df(time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.FWD:
            return direction * (spot - strike * df)
        else:
            self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)

    def calc_delta(self) -> float:
        return self.contract.direction * 1.0

    def calc_gamma(self) -> float:
        return 0.0

    def calc_vega(self) -> float:
        return 0.0

    def calc_theta(self) -> float:
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry - self.valuation_time
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.FWD:
            greek = -strike * df * rate
        else:
            self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)

        return self.contract.direction * greek

    def calc_rho(self) -> float:
        strike = self.contract.strike
        time_to_expiry = self.contract.expiry - self.valuation_time
        df = self.model.calc_df(time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.FWD:
            greek = strike * df * time_to_expiry
        else:
            self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)

        return self.contract.direction * greek


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def calc_d1(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        return (
            np.log(spot_over_strike) + (rate + 0.5 * vol ** 2) * time_to_expiry
        ) / (vol * np.sqrt(time_to_expiry))

    @staticmethod
    def calc_d2(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        d1 = EuropeanAnalyticPricer.calc_d1(spot_over_strike, vol, rate, time_to_expiry)
        return d1 - vol * np.sqrt(time_to_expiry)

    def __init__(self, contract: EuropeanContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, EuropeanContract):
            raise TypeError(
                f'Contract must be of type EuropeanContract but received {type(contract).__name__}'
            )
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.CALL:
            return direction * (spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
        elif self.contract.derivative_type == PutCallFwd.PUT:
            return direction * (strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
        else:
            self.contract.raise_incorrect_derivative_type_error()

    def calc_delta(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = norm.cdf(d1)
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -norm.cdf(-d1)
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_gamma(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)

        if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
            greek = norm.pdf(d1) / (spot * vol * np.sqrt(time_to_expiry))
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_vega(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)

        if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
            greek = strike * df * norm.pdf(d2) * np.sqrt(time_to_expiry)
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_theta(self) -> float:
        strike = self.contract.strike
        spot = self.model.spot
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = -((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                      + rate * strike * df * norm.cdf(d2))
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                      - rate * strike * df * norm.cdf(-d2))
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_rho(self) -> float:
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = strike * time_to_expiry * df * norm.cdf(d2)
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -strike * time_to_expiry * df * norm.cdf(-d2)
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek


class EuropeanDigitalAnalyticPricer(Pricer):
    def __init__(self, contract: EuropeanDigitalContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, EuropeanDigitalContract):
            raise TypeError(
                f'Contract must be of type EuropeanDigitalContract but received {type(contract).__name__}'
            )
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.CALL:
            value = df * norm.cdf(d2)
        elif self.contract.derivative_type == PutCallFwd.PUT:
            value = df * norm.cdf(-d2)
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return direction * value

    def calc_delta(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        base = df * norm.pdf(d2) / (spot * vol * np.sqrt(time_to_expiry))

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = base
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -base
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_gamma(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        base = -df * norm.pdf(d2) / (spot ** 2 * vol * np.sqrt(time_to_expiry)) * (
            1.0 + d2 / (vol * np.sqrt(time_to_expiry))
        )

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = base
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -base
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_vega(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        base = df * norm.pdf(d2) * d1 / vol

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = -base
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = base
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_theta(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        a = (rate - 0.5 * vol ** 2) / (vol * np.sqrt(time_to_expiry)) - d2 / (2.0 * time_to_expiry)

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = rate * df * norm.cdf(d2) - df * norm.pdf(d2) * a
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = rate * df * norm.cdf(-d2) + df * norm.pdf(d2) * a
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek

    def calc_rho(self) -> float:
        spot = self.model.spot
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        vol = self.model.vol
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
        base = df * norm.pdf(d2) * np.sqrt(time_to_expiry) / vol

        if self.contract.derivative_type == PutCallFwd.CALL:
            greek = -time_to_expiry * df * norm.cdf(d2) + base
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -time_to_expiry * df * norm.cdf(-d2) - base
        else:
            self.contract.raise_incorrect_derivative_type_error()

        return self.contract.direction * greek
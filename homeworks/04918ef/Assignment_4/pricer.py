from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import math
import numpy as np
from scipy.stats import norm
from statistics import mean

from enums import *
from contract import *
from model import *
from numerical_method import *


class Pricer(ABC):
    # Finite difference configuration
    fd_method: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL
    bump_type: BumpType = BumpType.RELATIVE
    bump_size: float = 0.01

    valuation_time: float = 0.0
    relative_bump_size: float = 0.01

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params = params

    @classmethod
    def create_pricer(cls, contract: Contract, model: MarketModel, params: Params) -> Pricer:
        instance = cls.__new__(cls)
        instance.__init__(contract, model, params)
        return instance

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    def _calc_bumped_value(self, spot_shift: float) -> float:
        model_bumped = copy.deepcopy(self.model)
        model_bumped.bump_spot(spot_shift)
        pricer_bumped = self.create_pricer(self.contract, model_bumped, self.params)
        return pricer_bumped.calc_fair_value()

    def calc_delta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))

        spot = self.model.spot
        h = self.bump_size if self.bump_type == BumpType.ABSOLUTE else self.bump_size * spot

        if self.fd_method == FiniteDifferenceMethod.FORWARD:
            return (self._calc_bumped_value(h) - self.calc_fair_value()) / h
        elif self.fd_method == FiniteDifferenceMethod.BACKWARD:
            return (self.calc_fair_value() - self._calc_bumped_value(-h)) / h
        elif self.fd_method == FiniteDifferenceMethod.CENTRAL:
            return (self._calc_bumped_value(h) - self._calc_bumped_value(-h)) / (2.0 * h)
        else:
            raise ValueError(f"Unsupported finite difference method: {self.fd_method}")

    def calc_gamma(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))

        spot = self.model.spot
        h = self.bump_size if self.bump_type == BumpType.ABSOLUTE else self.bump_size * spot

        if self.fd_method == FiniteDifferenceMethod.FORWARD:
            v_2up = self._calc_bumped_value(2.0 * h)
            v_up = self._calc_bumped_value(h)
            v_base = self.calc_fair_value()
            return (v_2up - 2.0 * v_up + v_base) / (h ** 2)
        elif self.fd_method == FiniteDifferenceMethod.BACKWARD:
            v_base = self.calc_fair_value()
            v_down = self._calc_bumped_value(-h)
            v_2down = self._calc_bumped_value(-2.0 * h)
            return (v_base - 2.0 * v_down + v_2down) / (h ** 2)
        elif self.fd_method == FiniteDifferenceMethod.CENTRAL:
            v_up = self._calc_bumped_value(h)
            v_base = self.calc_fair_value()
            v_down = self._calc_bumped_value(-h)
            return (v_up - 2.0 * v_base + v_down) / (h ** 2)
        else:
            raise ValueError(f"Unsupported finite difference method: {self.fd_method}")

    def calc_vega(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        strike = self.contract.strike
        expiry = self.contract.expiry
        vol = self.model.get_vol(strike, expiry)
        bump_size = self.relative_bump_size * vol
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self.model)
            model.bump_volgrid(b)
            bumped_pricer = self.create_pricer(self.contract, model, self.params)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2.0 * bump_size)

    def calc_theta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = 1.0 / 365.0
        bumped_pricer = copy.deepcopy(self)
        bumped_pricer.valuation_time += bump_size
        return (bumped_pricer.calc_fair_value() - self.calc_fair_value()) / bump_size

    def calc_rho(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.relative_bump_size * self.model.risk_free_rate
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self.model)
            model.bump_rate(b)
            bumped_pricer = self.create_pricer(self.contract, model, self.params)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2.0 * bump_size)

    def raise_unsupported_greek_method_error(
            self,
            method: str,
            supported: tuple[GreekMethod, ...] = tuple(_.value for _ in GreekMethod)) -> None:
        raise ValueError(f'Unsupported GreekMethod {method} for Pricer {type(self).__name__}. '
                         f'Supported methods are: {", ".join(supported)}')

    def raise_pricer_not_implemented_error(self) -> None:
        raise RuntimeError('The pricing of this type of contract has not been implemented yet.')


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def calc_d1(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        return 1.0 / (vol * np.sqrt(time_to_expiry)) * (np.log(spot_over_strike) + (rate + vol ** 2 / 2.0) * time_to_expiry)

    @staticmethod
    def calc_d2(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        return EuropeanAnalyticPricer.calc_d1(spot_over_strike, vol, rate, time_to_expiry) - vol * np.sqrt(time_to_expiry)

    def __init__(self, contract: EuropeanContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.get_vol(strike, expiry)
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

    def calc_delta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            spot = self.model.spot
            strike = self.contract.strike
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
            if self.contract.derivative_type == PutCallFwd.CALL:
                greek = norm.cdf(d1)
            elif self.contract.derivative_type == PutCallFwd.PUT:
                greek = -norm.cdf(-d1)
            else:
                self.contract.raise_incorrect_derivative_type_error()
            return self.contract.direction * greek
        elif method == GreekMethod.BUMP:
            return super().calc_delta(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            spot = self.model.spot
            strike = self.contract.strike
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
            if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = norm.pdf(d1) / (spot * vol * np.sqrt(time_to_expiry))
            else:
                self.contract.raise_incorrect_derivative_type_error()
            return self.contract.direction * greek
        elif method == GreekMethod.BUMP:
            return super().calc_gamma(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_vega(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            strike = self.contract.strike
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)
            spot = self.model.spot
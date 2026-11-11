from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import numpy as np
from scipy.stats import norm
from enums import *
from contract import *
from model import *


# TASK (HW4):
# Delta and gamma by finite differences, with a selectable difference scheme
# (forward / backward / central) and a selectable bump type (absolute / relative).
# The two choices are stored in the class variables `fd_method` and `bump_type`;
# their labels are the enums FiniteDifferenceMethod and BumpType (in enums.py).


class Params:
    pass


class Pricer(ABC):
    # Only for theta calculation via bump and revaluation
    valuation_time: float = 0.0

    # ---- settings for numerical (bump) greeks --------------------------------
    # Which difference quotient to use for delta and gamma.
    fd_method: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL
    # How the spot bump h is set:
    #   RELATIVE: h = relative_bump_size * spot   (e.g. 1% of spot)
    #   ABSOLUTE: h = absolute_bump_size          (e.g. 1 currency unit)
    bump_type: BumpType = BumpType.RELATIVE
    relative_bump_size: float = 0.01
    absolute_bump_size: float = 1.0

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params = params

    @classmethod
    def create_pricer(cls, contract: Contract, model: MarketModel, params: Params) -> Pricer:
        instance = cls(contract, model, params)
        return instance

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    # ---- helpers for spot bumping ----------------------------------------------
    def get_spot_bump_size(self) -> float:
        """The spot step h used in the difference quotients."""
        if self.bump_type == BumpType.RELATIVE:
            return self.relative_bump_size * self.model.spot
        elif self.bump_type == BumpType.ABSOLUTE:
            return self.absolute_bump_size
        else:
            raise ValueError(f'Unknown bump type: {self.bump_type}')

    def calc_fair_value_with_spot_shift(self, shift: float) -> float:
        """
        Fair value after moving the spot by `shift` (shift may be negative or 0).
        The model is copied, so the original pricer is left untouched.
        """
        if shift == 0.0:
            return self.calc_fair_value()
        model_to_bump = copy.deepcopy(self.model)
        model_to_bump.bump_spot(shift)
        bumped_pricer = self.create_pricer(self.contract, model_to_bump, self.params)
        return bumped_pricer.calc_fair_value()

    # ---- numerical delta and gamma ---------------------------------------------
    def calc_delta(self, method: GreekMethod) -> float:
        """
        delta = dV/dS, approximated with spot step h:
            forward : (V(S+h) - V(S))   / h        error O(h)
            backward: (V(S)   - V(S-h)) / h        error O(h)
            central : (V(S+h) - V(S-h)) / (2h)     error O(h^2)
        """
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        h = self.get_spot_bump_size()
        V = self.calc_fair_value_with_spot_shift   # short name: V(x) = price with spot moved by x

        if self.fd_method == FiniteDifferenceMethod.FORWARD:
            return (V(h) - V(0.0)) / h
        elif self.fd_method == FiniteDifferenceMethod.BACKWARD:
            return (V(0.0) - V(-h)) / h
        elif self.fd_method == FiniteDifferenceMethod.CENTRAL:
            return (V(h) - V(-h)) / (2 * h)
        else:
            raise ValueError(f'Unknown finite difference method: {self.fd_method}')

    def calc_gamma(self, method: GreekMethod) -> float:
        """
        gamma = d^2V/dS^2, approximated with spot step h:
            forward : (V(S+2h) - 2V(S+h) + V(S))    / h^2    error O(h)
            backward: (V(S)    - 2V(S-h) + V(S-2h)) / h^2    error O(h)
            central : (V(S+h)  - 2V(S)   + V(S-h))  / h^2    error O(h^2)
        """
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        h = self.get_spot_bump_size()
        V = self.calc_fair_value_with_spot_shift

        if self.fd_method == FiniteDifferenceMethod.FORWARD:
            return (V(2 * h) - 2 * V(h) + V(0.0)) / h ** 2
        elif self.fd_method == FiniteDifferenceMethod.BACKWARD:
            return (V(0.0) - 2 * V(-h) + V(-2 * h)) / h ** 2
        elif self.fd_method == FiniteDifferenceMethod.CENTRAL:
            return (V(h) - 2 * V(0.0) + V(-h)) / h ** 2
        else:
            raise ValueError(f'Unknown finite difference method: {self.fd_method}')

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
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def calc_theta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = 1.0 / 365.0
        bumped_pricer = copy.deepcopy(self)
        bumped_pricer.valuation_time += bump_size
        bumped_fair_value = bumped_pricer.calc_fair_value()
        del bumped_pricer
        return (bumped_fair_value - self.calc_fair_value()) / bump_size

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
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def raise_unsupported_greek_method_error(
            self,
            method: str,
            supported: tuple[GreekMethod, ...] = (_.value for _ in GreekMethod)) -> None:
        raise ValueError(f'Unsupported GreekMethod {method} for Pricer {type(self).__name__}. '
                         f'Supported methods are: {", ".join(supported)}')

    def raise_pricer_not_implemented_error(self) -> None:
        raise RuntimeError(f'The pricing of this type of contract has not been implemented yet.')


class ForwardAnalyticPricer(Pricer):
    supported_deriv_type: tuple[PutCallFwd, ...] = (PutCallFwd.FWD,)

    def __init__(self, contract: ForwardContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, ForwardContract):
            raise TypeError(f'Contract must be of type ForwardContract but received {type(contract).__name__}')
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

    def calc_delta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            return self.contract.direction * 1.0
        elif method == GreekMethod.BUMP:
            return super().calc_delta(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            return 0.0
        elif method == GreekMethod.BUMP:
            return super().calc_gamma(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_vega(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            return 0.0
        elif method == GreekMethod.BUMP:
            return super().calc_vega(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_theta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self.contract.strike
            time_to_expiry = self.contract.expiry - self.valuation_time
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)
            if self.contract.derivative_type == PutCallFwd.FWD:
                greek = -1.0 * strike * df * rate
            else:
                self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)
            return self.contract.direction * greek
        elif method == GreekMethod.BUMP:
            return super().calc_theta(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_rho(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self.contract.strike
            time_to_expiry = self.contract.expiry - self.valuation_time
            df = self.model.calc_df(time_to_expiry)
            if self.contract.derivative_type == PutCallFwd.FWD:
                greek = strike * df * time_to_expiry
            else:
                self.contract.raise_incorrect_derivative_type_error(self.supported_deriv_type)
            return self.contract.direction * greek
        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self.raise_unsupported_greek_method_error(method)


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def calc_d1(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        return 1 / (vol * np.sqrt(time_to_expiry)) * (np.log(spot_over_strike) + (rate + vol**2 / 2) * time_to_expiry)

    @staticmethod
    def calc_d2(spot_over_strike: float, vol: float, rate: float, time_to_expiry: float) -> float:
        d1 = EuropeanAnalyticPricer.calc_d1(spot_over_strike, vol, rate, time_to_expiry)
        return d1 - vol * np.sqrt(time_to_expiry)

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
            greek = 0.0
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
            greek = 0.0
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
            greek = 0.0
            spot = self.model.spot
            strike = self.contract.strike
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)
            d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
            if self.contract.derivative_type in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = strike * df * norm.pdf(d2) * np.sqrt(time_to_expiry)
            else:
                self.contract.raise_incorrect_derivative_type_error()
            return self.contract.direction * greek
        elif method == GreekMethod.BUMP:
            return super().calc_vega(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_theta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self.contract.strike
            spot = self.model.spot
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)
            d1 = EuropeanAnalyticPricer.calc_d1(spot / strike, vol, rate, time_to_expiry)
            d2 = EuropeanAnalyticPricer.calc_d2(spot / strike, vol, rate, time_to_expiry)
            if self.contract.derivative_type == PutCallFwd.CALL:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                                + rate * strike * df * norm.cdf(d2))
            elif self.contract.derivative_type == PutCallFwd.PUT:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                                - rate * strike * df * norm.cdf(-d2))
            else:
                self.contract.raise_incorrect_derivative_type_error()
            return self.contract.direction * greek
        elif method == GreekMethod.BUMP:
            return super().calc_theta(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_rho(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self.contract.strike
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            spot = self.model.spot
            vol = self.model.get_vol(strike, expiry)
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
        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self.raise_unsupported_greek_method_error(method)

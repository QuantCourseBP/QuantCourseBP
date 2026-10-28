from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import numpy as np
from scipy.stats import norm
from enums import *
from contract import *
from model import *


# TASK:
# Implement calc_delta() functions using finite difference approximation


class Params:
    pass

"""For the homework I add functions and three new variables to this class"""
# The finite difference formulas are from the 7th week's notebook
class Pricer(ABC):
    # Only for theta calculation via bump and revaluation
    valuation_time: float = 0.0
    relative_bump_size: float = 0.01
    # Three new variables:
    absolute_bump_size: float = 0.01
    bump_type: BumpSize = BumpSize.RELATIVE # by default the type is relative
    fdmethod: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params = params

    @classmethod
    def create_pricer(cls, contract: Contract, model: MarketModel, params: Params) -> Pricer:
        instance = cls(contract, model, params)
        return instance

    @abstractmethod # Every specific pricer created must implement its own version of it
    # for example ForwardAnalyticPricer, EuropeanAnalyticpricer
    def calc_fair_value(self) -> float:
        pass

    # Here come two new function related to bump size
    # del is for deleting temporary objects to save memory
    
    def get_bump_size(self) -> float:
        """Helps to determine the bump h based on BumpSize"""
        if self.bump_type == BumpSize.RELATIVE:
            return self.relative_bump_size * self.model.spot
        elif self.bump_type == BumpSize.ABSOLUTE:
            return self.absolute_bump_size
        else:
            raise ValueError(f"Unknown BumpSize: {self.bump_type}")
    
    # This is a generalization of a codepart used in calc_delta and calc_gamma in class  
    def fair_value_bump(self, bump: float) -> float:
        """Clones the model, bump spot by 'bump' (h), and revalue."""
        model_to_bump = copy.deepcopy(self.model)
        # Because of the deepcopy, only the cloned model's spot price changes.
        model_to_bump.bump_spot(bump) # It is defined in model.py MarketModel class
        bumped_pricer = self.create_pricer(self.contract, model_to_bump, self.params) # a new pricer object
        fv = bumped_pricer.calc_fair_value() # fv = fair value
        del bumped_pricer
        del model_to_bump
        return fv
        
    # I will modify calc_delta and calc_gamma 
    def calc_delta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        # Bump size:
        h = self.get_bump_size()
        
        if self.fdmethod == FiniteDifferenceMethod.CENTRAL:
            # (f(x+h)-f(x-h))/(2h)
            fv_up = self.fair_value_bump(h)
            fv_down = self.fair_value_bump(-h)
            return (fv_up - fv_down)/(2.0*h)
        elif self.fdmethod == FiniteDifferenceMethod.FORWARD:
            # (f(x+h)-f(x))/h
            fv_up = self.fair_value_bump(h)
            fv_base = self.calc_fair_value()
            return (fv_up - fv_base)/h
        elif self.fdmethod == FiniteDifferenceMethod.BACKWARD:
            # (f(x)-f(x-h))/h
            fv_base = self.calc_fair_value()
            fv_down = self.fair_value_bump(-h)
            return (fv_base - fv_down)/h
        else:
            raise ValueError(f"Unknown FiniteDifferenceMethod: {self.fdmethod}")

    def calc_gamma(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        # Bump size:
        h = self.get_bump_size()
        fv_base = self.calc_fair_value()
                
        if self.fdmethod == FiniteDifferenceMethod.CENTRAL:
            # (f(x+h)-2f(x)+f(x-h))/(h^2)
            fv_up = self.fair_value_bump(h)
            fv_down = self.fair_value_bump(-h)
            return (fv_up - 2.0*fv_base + fv_down)/(h**2)
        elif self.fdmethod == FiniteDifferenceMethod.FORWARD:
            # (f(x+2h)-2f(x+h)+f(x))/(h^2)
            fv_up_2 = self.fair_value_bump(2.0*h)
            fv_up_1 = self.fair_value_bump(h)
            return (fv_up_2 - 2.0*fv_up_1 + fv_base)/(h**2)
        elif self.fdmethod == FiniteDifferenceMethod.BACKWARD:
            # (f(x)-2f(x-h)+f(x-2h))/(h**2)
            fv_down_1 = self.fair_value_bump(-h)
            fv_down_2 = self.fair_value_bump(-2.0*h)
            return (fv_base - 2.0*fv_down_1 + fv_down_2)/(h**2)
        else:
            raise ValueError(f"Unknown FiniteDifferenceMethod: {self.fdmethod}")
                
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
            return super().calc_delta(method) # with super(). looks up calc_delta in the parent class (Pricer)
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

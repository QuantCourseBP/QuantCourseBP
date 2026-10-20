from __future__ import annotations
from abc import ABC, abstractmethod
from scipy.stats import norm
from src.enums import *
from contract import *
from model import *


# TASK:
# 1. Implement calc_fair_value() method in EuropeanAnalyticPricer.


class Params:
    pass


class Pricer(ABC):
    # Only for theta calculation via bump and revaluation
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

    def calc_delta(self) -> float:
        return self.contract.direction * 1.0

    def calc_gamma(self) -> float:
        return 0.0

    def calc_vega(self) -> float:
        return 0.0

    def calc_theta(self) -> float:
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

    def calc_rho(self) -> float:
        greek = 0.0
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
        greek = 0.0
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
        greek = 0.0
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
        greek = 0.0
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
        greek = 0.0
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
            greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                            + rate * strike * df * norm.cdf(d2))
        elif self.contract.derivative_type == PutCallFwd.PUT:
            greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                            - rate * strike * df * norm.cdf(-d2))
        else:
            self.contract.raise_incorrect_derivative_type_error()
        return self.contract.direction * greek

    def calc_rho(self) -> float:
        greek = 0.0
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



from pricer import Pricer, Params
from contract import EuropeanDigitalContract

class EuropeanDigitalAnalyticPricer(Pricer):

    def __init__(self, contract: EuropeanDigitalContract,
                 model: MarketModel, params: Params) -> None:

        # Check that the contract is a European digital option.
        if not isinstance(contract, EuropeanDigitalContract):
            raise TypeError("Contract must be EuropeanDigitalContract")

        # Initialize the parent Pricer class.
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:

        # Get the contract parameters.
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry

        # Calculate the remaining time to expiry.
        time_to_expiry = expiry - self.valuation_time

        # Get the market parameters.
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate

        # Calculate the discount factor.
        df = self.model.calc_df(time_to_expiry)

        # Calculate d2 using the Black-Scholes formula.
        d2 = EuropeanAnalyticPricer.calc_d2(
            spot / strike, vol, rate, time_to_expiry
        )

        # Calculate the fair value for digital CALL or PUT.
        if self.contract.derivative_type == PutCallFwd.CALL:
            fair_value = df * norm.cdf(d2)

        elif self.contract.derivative_type == PutCallFwd.PUT:
            fair_value = df * norm.cdf(-d2)

        else:
            self.contract.raise_incorrect_derivative_type_error()

        # Apply the direction of the position.
        return direction * fair_value

   
    def calc_delta(self) -> float:

        # Get the contract parameters.
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry

        # Calculate the remaining time to expiry.
        time_to_expiry = expiry - self.valuation_time

        # Get the market parameters.
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate

        # Calculate the discount factor and d2.
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(
            spot / strike, vol, rate, time_to_expiry
        )

        # Calculate the digital CALL delta.
        delta = df * norm.pdf(d2) / (
            spot * vol * np.sqrt(time_to_expiry)
        )

        # Digital PUT delta has the opposite sign.
        if self.contract.derivative_type == PutCallFwd.PUT:
            delta = -delta

        elif self.contract.derivative_type != PutCallFwd.CALL:
            self.contract.raise_incorrect_derivative_type_error()

        # Apply the LONG or SHORT position direction.
        return direction * delta



    def calc_gamma(self) -> float:

        # Get the contract parameters.
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry

        # Calculate the remaining time to expiry.
        time_to_expiry = expiry - self.valuation_time

        # Get the market parameters.
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate

        # Calculate the discount factor and d2.
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(
            spot / strike, vol, rate, time_to_expiry
        )

        # Calculate gamma for a digital CALL option.
        gamma = -df * norm.pdf(d2) / (
            spot**2 * vol * np.sqrt(time_to_expiry)
        ) * (1 + d2 / (vol * np.sqrt(time_to_expiry)))

        # Digital PUT gamma has the opposite sign.
        if self.contract.derivative_type == PutCallFwd.PUT:
            gamma = -gamma

        elif self.contract.derivative_type != PutCallFwd.CALL:
            self.contract.raise_incorrect_derivative_type_error()

        # Apply the position direction.
        return direction * gamma

 
    def calc_vega(self) -> float:

        # Get the contract parameters.
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry

        # Calculate the remaining time to expiry.
        time_to_expiry = expiry - self.valuation_time

        # Get the market parameters.
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate

        # Calculate the discount factor.
        df = self.model.calc_df(time_to_expiry)

        # Calculate d1 and d2 using the Black-Scholes formulas.
        d1 = EuropeanAnalyticPricer.calc_d1(
            spot / strike, vol, rate, time_to_expiry
        )

        d2 = EuropeanAnalyticPricer.calc_d2(
            spot / strike, vol, rate, time_to_expiry
        )

        # Calculate vega for a digital CALL option.
        vega = -df * norm.pdf(d2) * d1 / vol

        # Digital PUT has the opposite vega.
        if self.contract.derivative_type == PutCallFwd.PUT:
            vega = -vega

        elif self.contract.derivative_type != PutCallFwd.CALL:
            self.contract.raise_incorrect_derivative_type_error()

        # Apply the position direction.
        return direction * vega




    def calc_theta(self) -> float:

        # Get the contract parameters.
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry

        # Calculate the remaining time to expiry.
        time_to_expiry = expiry - self.valuation_time

        # Get the market parameters.
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate

        # Calculate the discount factor and d2.
        df = self.model.calc_df(time_to_expiry)

        d2 = EuropeanAnalyticPricer.calc_d2(
            spot / strike, vol, rate, time_to_expiry
        )

        # Calculate the derivative of d2 with respect to time to expiry.
        d2_derivative = (
            -np.log(spot / strike) /
            (2 * vol * time_to_expiry**1.5)
            + (rate - vol**2 / 2) /
            (2 * vol * np.sqrt(time_to_expiry))
        )

        # Calculate theta for a digital CALL.
        if self.contract.derivative_type == PutCallFwd.CALL:
            theta = (
                rate * df * norm.cdf(d2)
                - df * norm.pdf(d2) * d2_derivative
            )

        # Calculate theta for a digital PUT.
        elif self.contract.derivative_type == PutCallFwd.PUT:
            theta = (
                rate * df * norm.cdf(-d2)
                + df * norm.pdf(d2) * d2_derivative
            )

        else:
            self.contract.raise_incorrect_derivative_type_error()

        # Apply the position direction.
        return direction * theta


    def calc_rho(self) -> float:

        # Get the contract parameters.
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry

        # Calculate the remaining time to expiry.
        time_to_expiry = expiry - self.valuation_time

        # Get the market parameters.
        spot = self.model.spot
        vol = self.model.vol
        rate = self.model.risk_free_rate

        # Calculate the discount factor.
        df = self.model.calc_df(time_to_expiry)

        # Calculate d2 using the existing Black-Scholes method.
        d2 = EuropeanAnalyticPricer.calc_d2(
            spot / strike, vol, rate, time_to_expiry
        )

        # Calculate rho for a digital CALL option.
        if self.contract.derivative_type == PutCallFwd.CALL:
            rho = (
                -time_to_expiry * df * norm.cdf(d2)
                + df * norm.pdf(d2) * np.sqrt(time_to_expiry) / vol
            )

        # Calculate rho for a digital PUT option.
        elif self.contract.derivative_type == PutCallFwd.PUT:
            rho = (
                -time_to_expiry * df * norm.cdf(-d2)
                - df * norm.pdf(d2) * np.sqrt(time_to_expiry) / vol
            )

        else:
            self.contract.raise_incorrect_derivative_type_error()

        # Apply the position direction.
        return direction * rho



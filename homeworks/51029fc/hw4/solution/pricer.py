from __future__ import annotations
from abc import ABC, abstractmethod
from scipy.stats import norm
from src.enums import *
from contract import *
from model import *
from enum import Enum


class FiniteDifferenceMethod(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"


class BumpType(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
# TASK:
# 1. Implement calc_fair_value() method in EuropeanAnalyticPricer.


class Params:
    pass


class Pricer(ABC):
    # Only for theta calculation via bump and revaluation
    valuation_time: float = 0.0

    finite_difference_method = FiniteDifferenceMethod.CENTRAL
    bump_type = BumpType.RELATIVE

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params = params

    def _get_bump(self, bump_size: float) -> float:
        """Convert absolute or relative bump into a spot change."""

        if bump_size <= 0:
            raise ValueError("Bump size must be positive")

        spot = self.model.spot

        if self.bump_type == BumpType.ABSOLUTE:
            h = bump_size

        elif self.bump_type == BumpType.RELATIVE:
            h = spot * bump_size

        else:
            raise ValueError("Unknown bump type")

        if h <= 0 or spot + h == spot:
            raise ValueError("Bump is too small")

        return h

    def _price_at_spot(self, spot: float) -> float:
        """Calculate option price at a bumped spot."""

        if spot <= 0:
            raise ValueError("Spot must be positive")

        original_spot = self.model.spot

        try:
            self.model.spot = spot
            return self.calc_fair_value()

        finally:
            self.model.spot = original_spot

    def calc_delta_fd(self, bump_size: float = 1e-5) -> float:
        """Calculate delta using finite differences."""

        S = self.model.spot
        h = self._get_bump(bump_size)

        method = self.finite_difference_method

        if method == FiniteDifferenceMethod.FORWARD:
            return (
                self._price_at_spot(S + h)
                - self._price_at_spot(S)
            ) / h

        elif method == FiniteDifferenceMethod.BACKWARD:
            if S - h <= 0:
                raise ValueError("Bumped spot must be positive")

            return (
                self._price_at_spot(S)
                - self._price_at_spot(S - h)
            ) / h

        elif method == FiniteDifferenceMethod.CENTRAL:
            if S - h <= 0:
                raise ValueError("Bumped spot must be positive")

            return (
                self._price_at_spot(S + h)
                - self._price_at_spot(S - h)
            ) / (2 * h)

        raise ValueError("Unknown finite difference method")

    def calc_gamma_fd(self, bump_size: float = 1e-4) -> float:
        """Calculate gamma using finite differences."""

        S = self.model.spot
        h = self._get_bump(bump_size)

        method = self.finite_difference_method

        if method == FiniteDifferenceMethod.FORWARD:
            return (
                self._price_at_spot(S + 2*h)
                - 2*self._price_at_spot(S + h)
                + self._price_at_spot(S)
            ) / h**2

        elif method == FiniteDifferenceMethod.BACKWARD:
            if S - 2*h <= 0:
                raise ValueError("Bumped spot must be positive")

            return (
                self._price_at_spot(S)
                - 2*self._price_at_spot(S - h)
                + self._price_at_spot(S - 2*h)
            ) / h**2

        elif method == FiniteDifferenceMethod.CENTRAL:
            if S - h <= 0:
                raise ValueError("Bumped spot must be positive")

            return (
                self._price_at_spot(S + h)
                - 2*self._price_at_spot(S)
                + self._price_at_spot(S - h)
            ) / h**2

        raise ValueError("Unknown finite difference method")

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


class EuropeanDigitalAnalyticPricer(Pricer):

    def __init__(
        self,
        contract: EuropeanDigitalContract,
        model: MarketModel,
        params: Params
    ) -> None:
        if not isinstance(contract, EuropeanDigitalContract):
            raise TypeError("Expected EuropeanDigitalContract")

        super().__init__(contract, model, params)

    def _values(self):
        S = self.model.spot
        K = self.contract.strike
        T = self.contract.expiry - self.valuation_time
        r = self.model.risk_free_rate
        vol = self.model.vol

        if S <= 0 or K <= 0 or T <= 0 or vol <= 0:
            raise ValueError("Spot, strike, time and vol must be positive")

        d1 = EuropeanAnalyticPricer.calc_d1(S / K, vol, r, T)
        d2 = d1 - vol * np.sqrt(T)

        df = self.model.calc_df(T)

        if self.contract.derivative_type == PutCallFwd.CALL:
            eps = 1
        elif self.contract.derivative_type == PutCallFwd.PUT:
            eps = -1
        else:
            self.contract.raise_incorrect_derivative_type_error()

        direction = self.contract.direction

        return S, K, T, r, vol, d1, d2, df, eps, direction

    def calc_fair_value(self) -> float:
        S, K, T, r, vol, d1, d2, df, eps, direction = self._values()
        return float(direction * df * norm.cdf(eps * d2))

    def calc_delta(self) -> float:
        S, K, T, r, vol, d1, d2, df, eps, direction = self._values()
        return float(
            direction * eps * df * norm.pdf(d2)
            / (S * vol * np.sqrt(T))
        )

    def calc_gamma(self) -> float:
        S, K, T, r, vol, d1, d2, df, eps, direction = self._values()
        return float(
            -direction * eps * df * norm.pdf(d2)
            * (1 + d2 / (vol * np.sqrt(T)))
            / (S**2 * vol * np.sqrt(T))
        )

    def calc_vega(self) -> float:
        S, K, T, r, vol, d1, d2, df, eps, direction = self._values()
        return float(
            -direction * eps * df * norm.pdf(d2) * d1 / vol
        )

    def calc_theta(self) -> float:
        S, K, T, r, vol, d1, d2, df, eps, direction = self._values()

        dd2_dT = (
            (r - 0.5 * vol**2) / (2 * vol * np.sqrt(T))
            - np.log(S / K) / (2 * vol * T**1.5)
        )

        return float(
            direction * df * (
                r * norm.cdf(eps * d2)
                - eps * norm.pdf(d2) * dd2_dT
            )
        )

    def calc_rho(self) -> float:
        S, K, T, r, vol, d1, d2, df, eps, direction = self._values()
        return float(
            direction * T * df * (
                -norm.cdf(eps * d2)
                + eps * norm.pdf(d2) / (vol * np.sqrt(T))
            )
        )

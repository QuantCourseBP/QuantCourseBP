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


class Pricer(ABC):
    # Only for theta calculation via bump and revaluation
    valuation_time: float = 0.0

    # Bump-size settings
    relative_bump_size: float = 0.01
    absolute_bump_size: float = 0.01

    # Numerical delta/gamma settings
    finite_difference_method: FiniteDifferenceMethod = (
        FiniteDifferenceMethod.CENTRAL
    )

    bump_type: BumpType = BumpType.RELATIVE

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

    def _spot_bump_size(self) -> float:
        """
        Convert the configured bump into an absolute spot-price step.

        Absolute:
            h = absolute_bump_size

        Relative:
            h = relative_bump_size * spot
        """
        spot = float(self.model.spot)

        if not np.isfinite(spot) or spot <= 0:
            raise ValueError(
                "Spot must be finite and strictly positive."
            )

        bump_type = BumpType(self.bump_type)

        if bump_type == BumpType.ABSOLUTE:
            h = float(self.absolute_bump_size)
        else:
            h = float(self.relative_bump_size) * spot

        if not np.isfinite(h) or h <= 0:
            raise ValueError(
                "The effective spot bump must be finite and positive."
            )

        return h

    def _spot_values(
        self,
        offsets: tuple[int, ...],
        h: float
    ) -> dict[int, float]:
        """
        Calculate V(S + offset*h) for the requested offsets.

        Independent copies preserve the original pricer and model.
        Copying the entire pricer also preserves valuation_time.
        """
        spot = float(self.model.spot)

        with np.errstate(over="ignore", invalid="ignore"):
            sampled_spots = np.array([
                spot + offset * h
                for offset in offsets
            ])

        if (
            not np.all(np.isfinite(sampled_spots))
            or np.any(sampled_spots <= 0)
        ):
            raise ValueError(
                "Every bumped spot must be finite and positive. "
                "Reduce the bump size."
            )

        if len(np.unique(sampled_spots)) != len(sampled_spots):
            raise ValueError(
                "The bump is too small to create distinct "
                "floating-point spots."
            )

        values = {}

        for offset in offsets:
            if offset == 0:
                value = self.calc_fair_value()

            else:
                bumped_pricer = copy.deepcopy(self)

                bumped_pricer.model.bump_spot(
                    offset * h
                )

                value = bumped_pricer.calc_fair_value()

            if not np.isfinite(value):
                raise ValueError(
                    "A repriced value is not finite."
                )

            values[offset] = float(value)

        return values

    def _calc_spot_greek(self, order: int) -> float:
        """
        Calculate delta or gamma.

        order = 1: delta
        order = 2: gamma
        """
        if order not in (1, 2):
            raise ValueError(
                "Derivative order must be 1 or 2."
            )

        if isinstance(self.contract, EuropeanContract):
            if self.contract.expiry <= self.valuation_time:
                raise ValueError(
                    "European finite-difference Greeks require "
                    "positive time to expiry."
                )

        scheme = FiniteDifferenceMethod(
            self.finite_difference_method
        )

        h = self._spot_bump_size()

        with np.errstate(
            over="ignore",
            under="ignore",
            invalid="ignore"
        ):
            denominator = (
                h
                if order == 1
                else float(np.float64(h) * np.float64(h))
            )

        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError(
                "The finite-difference denominator "
                "underflows or overflows."
            )

        # Required sample locations.
        if order == 1:
            offsets = {
                FiniteDifferenceMethod.FORWARD: (0, 1),
                FiniteDifferenceMethod.BACKWARD: (-1, 0),
                FiniteDifferenceMethod.CENTRAL: (-1, 1),
            }[scheme]

        else:
            offsets = {
                FiniteDifferenceMethod.FORWARD: (0, 1, 2),
                FiniteDifferenceMethod.BACKWARD: (-2, -1, 0),
                FiniteDifferenceMethod.CENTRAL: (-1, 0, 1),
            }[scheme]

        v = self._spot_values(offsets, h)

        # First derivative: delta
        if order == 1:
            if scheme == FiniteDifferenceMethod.FORWARD:
                return (v[1] - v[0]) / h

            if scheme == FiniteDifferenceMethod.BACKWARD:
                return (v[0] - v[-1]) / h

            return ((v[1] - v[-1]) / h) / 2.0

        # Second derivative: gamma
        if scheme == FiniteDifferenceMethod.FORWARD:
            numerator = (
                (v[2] - v[1])
                - (v[1] - v[0])
            )

        elif scheme == FiniteDifferenceMethod.BACKWARD:
            numerator = (
                (v[0] - v[-1])
                - (v[-1] - v[-2])
            )

        else:
            numerator = (
                (v[1] - v[0])
                + (v[-1] - v[0])
            )

        return numerator / denominator

    def calc_delta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(
                method,
                (GreekMethod.BUMP,)
            )

        return self._calc_spot_greek(order=1)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(
                method,
                (GreekMethod.BUMP,)
            )

        return self._calc_spot_greek(order=2)

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

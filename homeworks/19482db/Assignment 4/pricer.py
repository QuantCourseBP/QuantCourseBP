from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import numpy as np
from scipy.stats import norm
from enums import *
from contract import *
from model import *


class Params:
    def __init__(self, bump_size: float = 1e-4):
        if bump_size <= 0:
            raise ValueError("Bump size must be positive.")
        self.bump_size = bump_size


class Pricer(ABC):
    # Finite difference configuration
    finite_difference_method: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL
    bump_size_type: BumpSizeType = BumpSizeType.RELATIVE

    # Only for theta calculation via bump and revaluation
    valuation_time: float = 0.0
    relative_bump_size: float = 0.01

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params = params

    @classmethod
    def create_pricer(cls, contract: Contract, model: MarketModel, params: Params) -> Pricer:
        return cls(contract, model, params)

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    def _get_bump_size(self) -> float:
        """Calculate spot bump using the configured bump convention."""
        bump = self.params.bump_size

        if bump <= 0:
            raise ValueError("Bump size must be positive.")

        if self.bump_size_type == BumpSizeType.ABSOLUTE:
            h = bump
        elif self.bump_size_type == BumpSizeType.RELATIVE:
            h = abs(self.model.spot) * bump
        else:
            raise ValueError(
                f"Unsupported bump size type: {self.bump_size_type}"
            )

        if h <= 0:
            raise ValueError("Calculated bump size must be positive.")

        return h

    def _get_bumped_fair_value(self, spot_bump: float) -> float:
        """Calculate fair value using a copied model with bumped spot."""
        model_to_bump = copy.deepcopy(self.model)
        model_to_bump.bump_spot(spot_bump)

        if model_to_bump.spot <= 0:
            raise ValueError("Bumped spot price must be positive.")

        bumped_pricer = self.create_pricer(
            self.contract,
            model_to_bump,
            self.params
        )
        bumped_pricer.valuation_time = self.valuation_time

        return bumped_pricer.calc_fair_value()

    def calc_delta(self, method: GreekMethod) -> float:
        """Calculate delta using forward, backward or central differences."""
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(
                method, (GreekMethod.BUMP,)
            )

        spot = self.model.spot
        h = self._get_bump_size()
        fd_method = self.finite_difference_method

        if fd_method == FiniteDifferenceMethod.FORWARD:
            v_0 = self.calc_fair_value()
            v_up = self._get_bumped_fair_value(h)
            return (v_up - v_0) / h

        elif fd_method == FiniteDifferenceMethod.BACKWARD:
            if spot - h <= 0:
                raise ValueError(
                    "Backward difference requires spot - bump > 0."
                )

            v_0 = self.calc_fair_value()
            v_down = self._get_bumped_fair_value(-h)
            return (v_0 - v_down) / h

        elif fd_method == FiniteDifferenceMethod.CENTRAL:
            if spot - h <= 0:
                raise ValueError(
                    "Central difference requires spot - bump > 0."
                )

            v_up = self._get_bumped_fair_value(h)
            v_down = self._get_bumped_fair_value(-h)
            return (v_up - v_down) / (2.0 * h)

        else:
            raise ValueError(
                f"Unsupported finite difference method: {fd_method}"
            )

    def calc_gamma(self, method: GreekMethod) -> float:
        """Calculate gamma using forward, backward or central differences."""
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(
                method, (GreekMethod.BUMP,)
            )

        spot = self.model.spot
        h = self._get_bump_size()
        fd_method = self.finite_difference_method

        v_0 = self.calc_fair_value()

        if fd_method == FiniteDifferenceMethod.FORWARD:
            v_up = self._get_bumped_fair_value(h)
            v_up2 = self._get_bumped_fair_value(2.0 * h)
            return (v_up2 - 2.0 * v_up + v_0) / (h ** 2)

        elif fd_method == FiniteDifferenceMethod.BACKWARD:
            if spot - 2.0 * h <= 0:
                raise ValueError(
                    "Backward gamma requires spot - 2 * bump > 0."
                )

            v_down = self._get_bumped_fair_value(-h)
            v_down2 = self._get_bumped_fair_value(-2.0 * h)
            return (v_0 - 2.0 * v_down + v_down2) / (h ** 2)

        elif fd_method == FiniteDifferenceMethod.CENTRAL:
            if spot - h <= 0:
                raise ValueError(
                    "Central gamma requires spot - bump > 0."
                )

            v_up = self._get_bumped_fair_value(h)
            v_down = self._get_bumped_fair_value(-h)
            return (v_up - 2.0 * v_0 + v_down) / (h ** 2)

        else:
            raise ValueError(
                f"Unsupported finite difference method: {fd_method}"
            )

    def calc_vega(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(
                method, (GreekMethod.BUMP,)
            )

        strike = self.contract.strike
        expiry = self.contract.expiry
        vol = self.model.get_vol(strike, expiry)
        bump_size = self.relative_bump_size * vol
        bumped_fair_values = []

        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self.model)
            model.bump_volgrid(b)

            bumped_pricer = self.create_pricer(
                self.contract, model, self.params
            )
            bumped_pricer.valuation_time = self.valuation_time
            bumped_fair_values.append(bumped_pricer.calc_fair_value())

        return (
            bumped_fair_values[0] - bumped_fair_values[1]
        ) / (2.0 * bump_size)

    def calc_theta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(
                method, (GreekMethod.BUMP,)
            )

        bump_size = 1.0 / 365.0
        bumped_pricer = copy.deepcopy(self)
        bumped_pricer.valuation_time += bump_size

        bumped_fair_value = bumped_pricer.calc_fair_value()

        return (
            bumped_fair_value - self.calc_fair_value()
        ) / bump_size

    def calc_rho(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(
                method, (GreekMethod.BUMP,)
            )

        bump_size = self.relative_bump_size * self.model.risk_free_rate
        bumped_fair_values = []

        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self.model)
            model.bump_rate(b)

            bumped_pricer = self.create_pricer(
                self.contract, model, self.params
            )
            bumped_pricer.valuation_time = self.valuation_time
            bumped_fair_values.append(bumped_pricer.calc_fair_value())

        return (
            bumped_fair_values[0] - bumped_fair_values[1]
        ) / (2.0 * bump_size)

    def raise_unsupported_greek_method_error(
            self,
            method: str,
            supported: tuple[GreekMethod, ...] = (_.value for _ in GreekMethod)) -> None:
        raise ValueError(
            f'Unsupported GreekMethod {method} for Pricer '
            f'{type(self).__name__}. '
            f'Supported methods are: {", ".join(supported)}'
        )

    def raise_pricer_not_implemented_error(self) -> None:
        raise RuntimeError(
            'The pricing of this type of contract has not been implemented yet.'
        )


class ForwardAnalyticPricer(Pricer):
    supported_deriv_type: tuple[PutCallFwd, ...] = (PutCallFwd.FWD,)

    def __init__(self, contract: ForwardContract, model: MarketModel, params: Params) -> None:
        if not isinstance(contract, ForwardContract):
            raise TypeError(
                f'Contract must be of type ForwardContract but received '
                f'{type(contract).__name__}'
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
            self.contract.raise_incorrect_derivative_type_error(
                self.supported_deriv_type
            )

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
            strike = self.contract.strike
            time_to_expiry = self.contract.expiry - self.valuation_time
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)

            if self.contract.derivative_type == PutCallFwd.FWD:
                greek = -strike * df * rate
            else:
                self.contract.raise_incorrect_derivative_type_error(
                    self.supported_deriv_type
                )

            return self.contract.direction * greek

        elif method == GreekMethod.BUMP:
            return super().calc_theta(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_rho(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            strike = self.contract.strike
            time_to_expiry = self.contract.expiry - self.valuation_time
            df = self.model.calc_df(time_to_expiry)

            if self.contract.derivative_type == PutCallFwd.FWD:
                greek = strike * df * time_to_expiry
            else:
                self.contract.raise_incorrect_derivative_type_error(
                    self.supported_deriv_type
                )

            return self.contract.direction * greek

        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self.raise_unsupported_greek_method_error(method)


class EuropeanAnalyticPricer(Pricer):

    @staticmethod
    def calc_d1(
        spot_over_strike: float,
        vol: float,
        rate: float,
        time_to_expiry: float
    ) -> float:
        return (
            1 / (vol * np.sqrt(time_to_expiry))
            * (
                np.log(spot_over_strike)
                + (rate + vol**2 / 2) * time_to_expiry
            )
        )

    @staticmethod
    def calc_d2(
        spot_over_strike: float,
        vol: float,
        rate: float,
        time_to_expiry: float
    ) -> float:
        d1 = EuropeanAnalyticPricer.calc_d1(
            spot_over_strike, vol, rate, time_to_expiry
        )
        return d1 - vol * np.sqrt(time_to_expiry)

    def __init__(
        self,
        contract: EuropeanContract,
        model: MarketModel,
        params: Params
    ) -> None:
        if not isinstance(contract, EuropeanContract):
            raise TypeError(
                f'Contract must be of type EuropeanContract but received '
                f'{type(contract).__name__}'
            )
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

        d1 = EuropeanAnalyticPricer.calc_d1(
            spot / strike, vol, rate, time_to_expiry
        )
        d2 = EuropeanAnalyticPricer.calc_d2(
            spot / strike, vol, rate, time_to_expiry
        )

        if self.contract.derivative_type == PutCallFwd.CALL:
            return direction * (
                spot * norm.cdf(d1)
                - strike * df * norm.cdf(d2)
            )

        elif self.contract.derivative_type == PutCallFwd.PUT:
            return direction * (
                strike * df * norm.cdf(-d2)
                - spot * norm.cdf(-d1)
            )

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

            d1 = EuropeanAnalyticPricer.calc_d1(
                spot / strike, vol, rate, time_to_expiry
            )

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

            d1 = EuropeanAnalyticPricer.calc_d1(
                spot / strike, vol, rate, time_to_expiry
            )

            if self.contract.derivative_type in (
                PutCallFwd.CALL, PutCallFwd.PUT
            ):
                greek = norm.pdf(d1) / (
                    spot * vol * np.sqrt(time_to_expiry)
                )
            else:
                self.contract.raise_incorrect_derivative_type_error()

            return self.contract.direction * greek

        elif method == GreekMethod.BUMP:
            return super().calc_gamma(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_vega(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            spot = self.model.spot
            strike = self.contract.strike
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)

            d2 = EuropeanAnalyticPricer.calc_d2(
                spot / strike, vol, rate, time_to_expiry
            )

            if self.contract.derivative_type in (
                PutCallFwd.CALL, PutCallFwd.PUT
            ):
                greek = (
                    strike * df * norm.pdf(d2)
                    * np.sqrt(time_to_expiry)
                )
            else:
                self.contract.raise_incorrect_derivative_type_error()

            return self.contract.direction * greek

        elif method == GreekMethod.BUMP:
            return super().calc_vega(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_theta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            strike = self.contract.strike
            spot = self.model.spot
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)

            d1 = EuropeanAnalyticPricer.calc_d1(
                spot / strike, vol, rate, time_to_expiry
            )
            d2 = EuropeanAnalyticPricer.calc_d2(
                spot / strike, vol, rate, time_to_expiry
            )

            if self.contract.derivative_type == PutCallFwd.CALL:
                greek = -1.0 * (
                    (spot * norm.pdf(d1) * vol)
                    / (2 * np.sqrt(time_to_expiry))
                    + rate * strike * df * norm.cdf(d2)
                )

            elif self.contract.derivative_type == PutCallFwd.PUT:
                greek = -1.0 * (
                    (spot * norm.pdf(d1) * vol)
                    / (2 * np.sqrt(time_to_expiry))
                    - rate * strike * df * norm.cdf(-d2)
                )

            else:
                self.contract.raise_incorrect_derivative_type_error()

            return self.contract.direction * greek

        elif method == GreekMethod.BUMP:
            return super().calc_theta(method)
        else:
            self.raise_unsupported_greek_method_error(method)

    def calc_rho(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            strike = self.contract.strike
            expiry = self.contract.expiry
            time_to_expiry = expiry - self.valuation_time
            spot = self.model.spot
            vol = self.model.get_vol(strike, expiry)
            rate = self.model.risk_free_rate
            df = self.model.calc_df(time_to_expiry)

            d2 = EuropeanAnalyticPricer.calc_d2(
                spot / strike, vol, rate, time_to_expiry
            )

            if self.contract.derivative_type == PutCallFwd.CALL:
                greek = (
                    strike * time_to_expiry * df * norm.cdf(d2)
                )

            elif self.contract.derivative_type == PutCallFwd.PUT:
                greek = (
                    -strike * time_to_expiry * df * norm.cdf(-d2)
                )

            else:
                self.contract.raise_incorrect_derivative_type_error()

            return self.contract.direction * greek

        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self.raise_unsupported_greek_method_error(method)

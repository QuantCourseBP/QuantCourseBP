from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from src.enums import *
from src.contract import *
from src.model import *
from src.numerical_method import *
from scipy.stats import norm


class Pricer(ABC):
    RELATIVE_BUMP_SIZE: float = 0.01

    def __init__(self, contract: Contract, model: MarketModel, method: NumericalMethod):
        self._contract = contract
        self._model = model
        self._method = method

    @staticmethod
    def get_pricers() -> dict[str, Pricer]:
        return {cls.__name__: cls for cls in Pricer.__subclasses__()}

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    def calc_delta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.get_initial_spot()
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self._model)
            model.bump_initial_spot(b)
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._method)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.get_initial_spot()
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self._model)
            model.bump_initial_spot(b)
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._method)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - 2 * self.calc_fair_value() + bumped_fair_values[1]) / (bump_size ** 2)

    def calc_vega(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        spot = self._model.get_initial_spot()
        strike = self._contract.get_strike()
        tenor = self._contract.get_expiry()
        vol = self._model.get_volgrid().get_vol(tenor, spot / strike)
        bump_size = self.RELATIVE_BUMP_SIZE * vol
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self._model)
            model.bump_volgrid(b)
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._method)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def calc_theta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = 1.0 / 252.0
        contract = copy.deepcopy(self._contract)
        contract.set_expiry(self._contract.get_expiry() + bump_size)
        bumped_pricer = globals()[type(self).__name__](contract, self._model, self._method)
        bumped_fair_value = bumped_pricer.calc_fair_value()
        del bumped_pricer
        del contract
        return -1.0 * (bumped_fair_value - self.calc_fair_value()) / bump_size

    def calc_rho(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.get_rate()
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self._model)
            model.bump_rate(b)
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._method)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def _raise_unsupported_greek_method_error(
            self,
            method: str,
            supported: tuple[GreekMethod] = (x.value for x in GreekMethod)) -> None:
        raise ValueError(f'Unsupported GreekMethod {method} for Pricer {type(self).__name__}. '
                         f'Supported methods are: {", ".join(supported)}')


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def __d1(spot: float, strike: float, vol: float, rate: float, tenor: float):
        return 1 / (vol * np.sqrt(tenor)) * (np.log(spot / strike) + (rate + vol**2 / 2) * tenor)

    @staticmethod
    def __d2(spot: float, strike: float, vol: float, rate: float, tenor: float):
        d1 = EuropeanAnalyticPricer.__d1(spot, strike, vol, rate, tenor)
        return d1 - vol * np.sqrt(tenor)

    def __init__(self, contract: EuropeanContract, model: MarketModel, method: AnalyticMethod):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        if not isinstance(method, AnalyticMethod):
            raise TypeError(f'Method must be of type AnalyticMethod but received {type(method).__name__}')
        super().__init__(contract, model, method)

    def calc_fair_value(self) -> float:
        direction = self._contract.get_direction()
        strike = self._contract.get_strike()
        tenor = self._contract.get_expiry()
        if tenor < 1e-8:
            return 0.0  # return zero if option expired
        spot = self._model.get_initial_spot()
        vol = self._model.get_vol(tenor, spot / strike)
        rate = self._model.get_rate()
        df = self._model.get_df(tenor)
        d1 = EuropeanAnalyticPricer.__d1(spot, strike, vol, rate, tenor)
        d2 = EuropeanAnalyticPricer.__d2(spot, strike, vol, rate, tenor)
        if self._contract.get_type() == PutCallFwd.CALL:
            return direction * (spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
        elif self._contract.get_type() == PutCallFwd.PUT:
            return direction * (strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
        else:
            self.__raise_incorrect_derivative_type_error()

    def calc_delta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            spot = self._model.get_initial_spot()
            strike = self._contract.get_strike()
            tenor = self._contract.get_expiry()
            vol = self._model.get_volgrid().get_vol(tenor, spot / strike)
            rate = self._model.get_rate()
            d1 = EuropeanAnalyticPricer.__d1(spot, strike, vol, rate, tenor)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = norm.cdf(d1)
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -norm.cdf(-d1)
            else:
                self.__raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_delta(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            spot = self._model.get_initial_spot()
            strike = self._contract.get_strike()
            tenor = self._contract.get_expiry()
            vol = self._model.get_volgrid().get_vol(tenor, spot / strike)
            rate = self._model.get_rate()
            d1 = EuropeanAnalyticPricer.__d1(spot, strike, vol, rate, tenor)
            if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = norm.pdf(d1) / (spot * vol * np.sqrt(tenor))
            else:
                self.__raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_gamma(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_vega(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            spot = self._model.get_initial_spot()
            strike = self._contract.get_strike()
            tenor = self._contract.get_expiry()
            vol = self._model.get_volgrid().get_vol(tenor, spot / strike)
            rate = self._model.get_rate()
            df = self._model.get_df(tenor)
            d2 = EuropeanAnalyticPricer.__d2(spot, strike, vol, rate, tenor)
            if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = strike * df * norm.pdf(d2) * np.sqrt(tenor)
            else:
                self.__raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_vega(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_theta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self._contract.get_strike()
            spot = self._model.get_initial_spot()
            tenor = self._contract.get_expiry()
            vol = self._model.get_volgrid().get_vol(tenor, spot / strike)
            rate = self._model.get_rate()
            df = self._model.get_df(tenor)
            d1 = EuropeanAnalyticPricer.__d1(spot, strike, vol, rate, tenor)
            d2 = EuropeanAnalyticPricer.__d2(spot, strike, vol, rate, tenor)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(tenor)) + rate * strike * df * norm.cdf(d2))
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(tenor)) - rate * strike * df * norm.cdf(-d2))
            else:
                self.__raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_theta(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_rho(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self._contract.get_strike()
            tenor = self._contract.get_expiry()
            spot = self._model.get_initial_spot()
            vol = self._model.get_volgrid().get_vol(tenor, spot / strike)
            rate = self._model.get_rate()
            df = self._model.get_df(tenor)
            d2 = EuropeanAnalyticPricer.__d2(spot, strike, vol, rate, tenor)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = strike * tenor * df * norm.cdf(d2)
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -strike * tenor * df * norm.cdf(-d2)
            else:
                self.__raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def __raise_incorrect_derivative_type_error(self) -> None:
        raise ValueError(f'Derivative type of {type(self._contract).__name__} must be CALL or PUT')


# todo: to be implemented
class GenericTreePricer(Pricer):
    pass


# todo: to be implemented
class GenericPDEPricer(Pricer):
    pass


# todo: to be implemented
class GenericMCPricer(Pricer):
    pass

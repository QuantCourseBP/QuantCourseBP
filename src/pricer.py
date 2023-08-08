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
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        vol = self._model.get_vol(strike, expiry)
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
            supported: tuple[GreekMethod] = (_.value for _ in GreekMethod)) -> None:
        raise ValueError(f'Unsupported GreekMethod {method} for Pricer {type(self).__name__}. '
                         f'Supported methods are: {", ".join(supported)}')


class ForwardAnalyticPricer(Pricer):
    __supported_deriv_type: tuple = (PutCallFwd.FWD,)

    def __init__(self, contract: ForwardContract, model: MarketModel, method: AnalyticMethod):
        if not isinstance(contract, ForwardContract):
            raise TypeError(f'Contract must be of type ForwardContract but received {type(contract).__name__}')
        if not isinstance(method, AnalyticMethod):
            raise TypeError(f'Method must be of type AnalyticMethod but received {type(method).__name__}')
        super().__init__(contract, model, method)

    def calc_fair_value(self) -> float:
        direction = self._contract.get_direction()
        spot = self._model.get_initial_spot()
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        df = self._model.get_df(expiry)
        if self._contract.get_type() == PutCallFwd.FWD:
            return direction * (spot - strike * df)
        else:
            self._contract.raise_incorrect_derivative_type_error(self.__supported_deriv_type)

    def calc_delta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            return 1.0
        elif method == GreekMethod.BUMP:
            return super().calc_delta(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            return 0.0
        elif method == GreekMethod.BUMP:
            return super().calc_gamma(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_vega(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            return 0.0
        elif method == GreekMethod.BUMP:
            return super().calc_vega(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_theta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self._contract.get_strike()
            expiry = self._contract.get_expiry()
            rate = self._model.get_rate()
            df = self._model.get_df(expiry)
            if self._contract.get_type() == PutCallFwd.FWD:
                greek = -1.0 * strike * df * rate
            else:
                self._contract.raise_incorrect_derivative_type_error(self.__supported_deriv_type)
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_theta(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_rho(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self._contract.get_strike()
            expiry = self._contract.get_expiry()
            df = self._model.get_df(expiry)
            if self._contract.get_type() == PutCallFwd.FWD:
                greek = strike * df * expiry
            else:
                self._contract.raise_incorrect_derivative_type_error(self.__supported_deriv_type)
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self._raise_unsupported_greek_method_error(method)


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def d1(spot: float, strike: float, vol: float, rate: float, tenor: float):
        return 1 / (vol * np.sqrt(tenor)) * (np.log(spot / strike) + (rate + vol**2 / 2) * tenor)

    @staticmethod
    def d2(spot: float, strike: float, vol: float, rate: float, tenor: float):
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, tenor)
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
        expiry = self._contract.get_expiry()
        spot = self._model.get_initial_spot()
        vol = self._model.get_vol(strike, expiry)
        rate = self._model.get_rate()
        df = self._model.get_df(expiry)
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, expiry)
        d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, expiry)
        if self._contract.get_type() == PutCallFwd.CALL:
            return direction * (spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
        elif self._contract.get_type() == PutCallFwd.PUT:
            return direction * (strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
        else:
            self._contract.raise_incorrect_derivative_type_error()

    def calc_delta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            spot = self._model.get_initial_spot()
            strike = self._contract.get_strike()
            expiry = self._contract.get_expiry()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, expiry)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = norm.cdf(d1)
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -norm.cdf(-d1)
            else:
                self._contract.raise_incorrect_derivative_type_error()
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
            expiry = self._contract.get_expiry()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, expiry)
            if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = norm.pdf(d1) / (spot * vol * np.sqrt(expiry))
            else:
                self._contract.raise_incorrect_derivative_type_error()
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
            expiry = self._contract.get_expiry()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            df = self._model.get_df(expiry)
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, expiry)
            if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = strike * df * norm.pdf(d2) * np.sqrt(expiry)
            else:
                self._contract.raise_incorrect_derivative_type_error()
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
            expiry = self._contract.get_expiry()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            df = self._model.get_df(expiry)
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, expiry)
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, expiry)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(expiry))
                                + rate * strike * df * norm.cdf(d2))
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(expiry))
                                - rate * strike * df * norm.cdf(-d2))
            else:
                self._contract.raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_theta(method)
        else:
            self._raise_unsupported_greek_method_error(method)

    def calc_rho(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            strike = self._contract.get_strike()
            expiry = self._contract.get_expiry()
            spot = self._model.get_initial_spot()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            df = self._model.get_df(expiry)
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, expiry)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = strike * expiry * df * norm.cdf(d2)
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -strike * expiry * df * norm.cdf(-d2)
            else:
                self._contract.raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self._raise_unsupported_greek_method_error(method)


class GenericTreePricer(Pricer):
    def __init__(self, contract: Contract, model: MarketModel, params: TreeParams):
        self._contract = contract
        self._model = model
        self._params = params
        if np.isnan(params.up_step_mult) or np.isnan(params.down_step_mult):
            tree_method = BalancedSimpleBinomialTree(params, model)
        else:
            tree_method = SimpleBinomialTree(params, model)
        self._tree_method = tree_method

    def calc_fair_value(self) -> float:
        self._tree_method.init_tree()
        spot_tree = self._tree_method._spot_tree
        price_tree = [[np.nan for _ in level] for level in spot_tree]
        for i in range(len(spot_tree[-1])):
            log_spot = spot_tree[-1][i]
            spot = {self._contract.get_timeline()[0]: np.exp(log_spot)}
            discounted_price = self._tree_method._df[-1] * self._contract.payoff(spot)
            price_tree[-1][i] = discounted_price
        for step in range(self._params.nr_steps - 1, -1, -1):
            for i in range(len(spot_tree[step])):
                # discounted price is martingale
                discounted_price = self._tree_method._prob[0] * price_tree[step + 1][i] + \
                                   self._tree_method._prob[1] * price_tree[step + 1][i + 1]
                price_tree[step][i] = discounted_price
        return price_tree[0][0]
    
    def calc_delta(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_gamma(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_vega(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_theta(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')

    def calc_rho(self, method: GreekMethod) -> float:
        raise NotImplementedError('Greeks are not implemented for tree method yet.')


# todo: to be implemented
class GenericPDEPricer(Pricer):
    def calc_fair_value(self) -> float:
        raise NotImplementedError('Fair value is not implemented yet for GenericPDEPricer.')


# todo: to be implemented
class GenericMCPricer(Pricer):
    def calc_fair_value(self) -> float:
        raise NotImplementedError('Fair value is not implemented yet for GenericMCPricer.')

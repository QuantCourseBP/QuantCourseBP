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
        tenor = self._contract.get_expiry()
        if tenor < 1e-8:
            return 0.0  # return zero if option expired
        spot = self._model.get_initial_spot()
        vol = self._model.get_vol(tenor, spot / strike)
        rate = self._model.get_rate()
        df = self._model.get_df(tenor)
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, tenor)
        d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, tenor)
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
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, tenor)
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
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, tenor)
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
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, tenor)
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
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, tenor)
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, tenor)
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
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, tenor)
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


class GenericTreePricer(Pricer):
    def __init__(self, contract: VanillaOptionContract, model: FlatVol, params: TreeParams):
        self._contract = contract
        self._model = model
        self._params = params
        if (np.isnan(params._up_step_mult) or np.isnan(params._down_step_mult)):
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
            discounted_price = self._tree_method._df[-1] * self._contract.payoff(np.exp(log_spot))
            price_tree[-1][i] = discounted_price
        for step in range(self._params._nr_steps - 1,-1,-1):
            for i in range(len(spot_tree[step])):
                log_spot = spot_tree[step][i]
                # discounted price is martingale
                discounted_price = self._tree_method._prob[0] * price_tree[step + 1][i] + self._tree_method._prob[1] * price_tree[step + 1][i + 1]
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

<<<<<<< HEAD
=======

# todo: to be implemented
class GenericPDEPricer(Pricer):
    def calc_fair_value(self) -> float:
        raise NotImplementedError('Fair value is not implemented yet for GenericPDEPricer.')
>>>>>>> c7420b0e9198cc394b16a313520690abda82bcbc

class GenericPDEPricer(Pricer):
    def __init__(self, contract: EuropeanContract, model: BSVolModel, params: PDEParams):
        self._derivative_type = contract.get_type()
        self._bsPDE = BlackScholesPDE(model, params)
        self.grid = self._bsPDE.grid
        self._initial_spot = model.get_initial_spot()
        self._strike = contract.get_strike()
        self._interest_rate = model.get_rate()
        self.nt_steps = params.nt_steps
        self.ns_steps = params.ns_steps
        self.und_step = self._initial_spot / float(self.ns_steps)  # Number of time steps
        self.t_step = params.exp / float(self.nt_steps)  # Number of stock price steps

    def setup_boundary_conditions(self):
        if self._derivative_type == "CALL":
            self.grid[0, :] = np.maximum(np.linspace(0, self._initial_spot + self.ns_steps * self.und_step, self.ns_steps + 1) - self._strike, 0)
            self.grid[:, -1] = (self._initial_spot + self.ns_steps * self.und_step - self._strike) * np.exp(
                -self._interest_rate * self.t_step * (self.nt_steps - np.arange(self.nt_steps + 1)))

        else:
            self.grid[0, :] = np.maximum(self._strike - np.linspace(0, self._initial_spot + self.ns_steps * self.und_step, self.ns_steps + 1), 0)
            self.grid[:, -1] = (self._strike - self._initial_spot - self.ns_steps * self.und_step) * np.exp(
                -self._interest_rate * self.t_step * (self.nt_steps - np.arange(self.nt_steps  + 1)))

    def calc_fair_value(self, method=PDEMethod.EXPLICIT) -> float:
        self.setup_boundary_conditions()

        if method.upper() == PDEMethod.EXPLICIT:
            self._bsPDE.explicit_method()
        elif method.upper() == PDEMethod.IMPLICIT:
            self._bsPDE.implicit_method()
        elif method.upper() == PDEMethod.CRANKNICOLSON:
            self._bsPDE.crank_nicolson_method()
        else:
            raise ValueError("Invalid method. Use 'explicit', 'implicit', or 'crank_nicolson'.")

        return self.grid[0, self.ns_steps // 2]  # Return the option price at S0

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
class GenericMCPricer(Pricer):
    def calc_fair_value(self) -> float:
        raise NotImplementedError('Fair value is not implemented yet for GenericMCPricer.')

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import numpy as np
from scipy.stats import norm
from src.enums import *
from src.contract import *
from src.model import *
from src.numerical_method import *


class Pricer(ABC):
    RELATIVE_BUMP_SIZE: float = 0.01
    # only for theta calculation via bump and revaluation
    __valuation_time: float = 0.0

    def __init__(self, contract: Contract, model: MarketModel, params: Params):
        self._contract: Contract = contract
        self._model: MarketModel = model
        self._params: Params | MCParams | PDEParams | TreeParams = params

    @staticmethod
    def get_pricers() -> dict[str, Pricer]:
        return {cls.__name__: cls for cls in Pricer.__subclasses__()}

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    def calc_delta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.get_spot()
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self._model)
            model.bump_spot(b)
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._params)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.get_spot()
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self._model)
            model.bump_spot(b)
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._params)
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
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._params)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def calc_theta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = 1.0 / 365.0
        bumped_pricer = copy.deepcopy(self)
        bumped_pricer.__valuation_time += bump_size
        bumped_fair_value = bumped_pricer.calc_fair_value()
        del bumped_pricer
        return (bumped_fair_value - self.calc_fair_value()) / bump_size

    def calc_rho(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self._raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.get_rate()
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self._model)
            model.bump_rate(b)
            bumped_pricer = globals()[type(self).__name__](self._contract, model, self._params)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def _get_valuation_time(self) -> float:
        return self.__valuation_time

    def _raise_unsupported_greek_method_error(
            self,
            method: str,
            supported: tuple[GreekMethod] = (_.value for _ in GreekMethod)) -> None:
        raise ValueError(f'Unsupported GreekMethod {method} for Pricer {type(self).__name__}. '
                         f'Supported methods are: {", ".join(supported)}')


class ForwardAnalyticPricer(Pricer):
    __supported_deriv_type: tuple[PutCallFwd, ...] = (PutCallFwd.FWD,)

    def __init__(self, contract: ForwardContract, model: MarketModel, params: Params):
        if not isinstance(contract, ForwardContract):
            raise TypeError(f'Contract must be of type ForwardContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self._contract.get_direction()
        spot = self._model.get_spot()
        strike = self._contract.get_strike()
        time_to_expiry = self._contract.get_expiry() - self._get_valuation_time()
        df = self._model.get_df(time_to_expiry)
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
            time_to_expiry = self._contract.get_expiry() - self._get_valuation_time()
            rate = self._model.get_rate()
            df = self._model.get_df(time_to_expiry)
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
            time_to_expiry = self._contract.get_expiry() - self._get_valuation_time()
            df = self._model.get_df(time_to_expiry)
            if self._contract.get_type() == PutCallFwd.FWD:
                greek = strike * df * time_to_expiry
            else:
                self._contract.raise_incorrect_derivative_type_error(self.__supported_deriv_type)
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self._raise_unsupported_greek_method_error(method)


class EuropeanAnalyticPricer(Pricer):
    @staticmethod
    def d1(spot: float, strike: float, vol: float, rate: float, time_to_expiry: float):
        return 1 / (vol * np.sqrt(time_to_expiry)) * (np.log(spot / strike) + (rate + vol**2 / 2) * time_to_expiry)

    @staticmethod
    def d2(spot: float, strike: float, vol: float, rate: float, time_to_expiry: float):
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
        return d1 - vol * np.sqrt(time_to_expiry)

    def __init__(self, contract: EuropeanContract, model: MarketModel, params: Params):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self._contract.get_direction()
        strike = self._contract.get_strike()
        expiry = self._contract.get_expiry()
        time_to_expiry = expiry - self._get_valuation_time()
        spot = self._model.get_spot()
        vol = self._model.get_vol(strike, expiry)
        rate = self._model.get_rate()
        df = self._model.get_df(time_to_expiry)
        d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
        d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
        if self._contract.get_type() == PutCallFwd.CALL:
            return direction * (spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
        elif self._contract.get_type() == PutCallFwd.PUT:
            return direction * (strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1))
        else:
            self._contract.raise_incorrect_derivative_type_error()

    def calc_delta(self, method: GreekMethod) -> float:
        if method == GreekMethod.ANALYTIC:
            greek = 0.0
            spot = self._model.get_spot()
            strike = self._contract.get_strike()
            expiry = self._contract.get_expiry()
            time_to_expiry = expiry - self._get_valuation_time()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
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
            spot = self._model.get_spot()
            strike = self._contract.get_strike()
            expiry = self._contract.get_expiry()
            time_to_expiry = expiry - self._get_valuation_time()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
            if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = norm.pdf(d1) / (spot * vol * np.sqrt(time_to_expiry))
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
            spot = self._model.get_spot()
            strike = self._contract.get_strike()
            expiry = self._contract.get_expiry()
            time_to_expiry = expiry - self._get_valuation_time()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            df = self._model.get_df(time_to_expiry)
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
            if self._contract.get_type() in (PutCallFwd.CALL, PutCallFwd.PUT):
                greek = strike * df * norm.pdf(d2) * np.sqrt(time_to_expiry)
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
            spot = self._model.get_spot()
            expiry = self._contract.get_expiry()
            time_to_expiry = expiry - self._get_valuation_time()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            df = self._model.get_df(time_to_expiry)
            d1 = EuropeanAnalyticPricer.d1(spot, strike, vol, rate, time_to_expiry)
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
                                + rate * strike * df * norm.cdf(d2))
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -1.0 * ((spot * norm.pdf(d1) * vol) / (2 * np.sqrt(time_to_expiry))
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
            time_to_expiry = expiry - self._get_valuation_time()
            spot = self._model.get_spot()
            vol = self._model.get_vol(strike, expiry)
            rate = self._model.get_rate()
            df = self._model.get_df(time_to_expiry)
            d2 = EuropeanAnalyticPricer.d2(spot, strike, vol, rate, time_to_expiry)
            if self._contract.get_type() == PutCallFwd.CALL:
                greek = strike * time_to_expiry * df * norm.cdf(d2)
            elif self._contract.get_type() == PutCallFwd.PUT:
                greek = -strike * time_to_expiry * df * norm.cdf(-d2)
            else:
                self._contract.raise_incorrect_derivative_type_error()
            return self._contract.get_direction() * greek
        elif method == GreekMethod.BUMP:
            return super().calc_rho(method)
        else:
            self._raise_unsupported_greek_method_error(method)


class GenericTreePricer(Pricer):
    def __init__(self, contract: EuropeanContract, model: MarketModel, params: TreeParams):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        if not isinstance(params, TreeParams):
            raise TypeError(f'Params must be of type TreeParams but received {type(params).__name__}')
        super().__init__(contract, model, params)
        if np.isnan(self._params.up_step_mult) or np.isnan(self._params.down_step_mult):
            tree_method = BalancedSimpleBinomialTree(self._contract, self._model, self._params)
        else:
            tree_method = SimpleBinomialTree(self._contract, self._model, self._params)
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


class BlackScholesPDEPricer(Pricer):
    def __init__(self, contract: EuropeanContract, model: MarketModel, params: PDEParams):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        if not isinstance(params, PDEParams):
            raise TypeError(f'Params must be of type PDEParams but received {type(params).__name__}')
        self._derivative_type = contract.get_type()
        self._bsPDE = BlackScholesPDE(contract, model, params)
        self.grid = self._bsPDE.grid
        self._initial_spot = model.get_spot()
        self._strike = contract.get_strike()
        self._interest_rate = model.get_rate()
        self.t_step = params.time_step
        self.und_step = params.und_step
        self.stock_min = params.stock_min
        self.stock_max = params.stock_max
        self.ns_steps = self._bsPDE.num_of_und_steps  # Number of stock price steps
        self.nt_steps = self._bsPDE.num_of_time_steps  # Number of time steps
        self.method = params.method
        self.setup_boundary_conditions = self._bsPDE.setup_boundary_conditions()

    def calc_fair_value(self) -> float:

        if self.method.upper() == BSPDEMethod.EXPLICIT:
            self._bsPDE.explicit_method()
        elif self.method.upper() == BSPDEMethod.IMPLICIT:
            self._bsPDE.implicit_method()
        elif self.method.upper() == BSPDEMethod.CRANKNICOLSON:
            self._bsPDE.crank_nicolson_method()
        else:
            raise ValueError("Invalid method. Use 'explicit', 'implicit', or 'crank_nicolson'.")

        # linear interpolation
        down = int(np.floor((self._initial_spot - self.stock_min)/self.und_step))
        up = int(np.ceil((self._initial_spot - self.stock_min)/self.und_step))

        if down == up:
            return self.grid[1, down+1]
        else:
            return self.grid[1,  down+1] + (self.grid[1, up+1] - self.grid[1, down+1]) * \
                   (self._initial_spot - self.stock_min - down*self.und_step)/self.und_step


class GenericMCPricer(Pricer):
    def __init__(self, contract: Contract, model: MarketModel, params: MCParams):
        super().__init__(contract, model, params)
        self._mc_method = MCMethod(self._contract, self._model, self._params)

    def calc_fair_value(self) -> float:
        contract = self._contract
        contractual_timeline = contract.get_timeline()
        spot_paths = self._mc_method.simulate_spot_paths()
        num_of_paths = self._params.num_of_paths
        path_payoff = np.empty(num_of_paths)
        for path in range(num_of_paths):
            fixing_schedule = dict(zip(contractual_timeline, spot_paths[path, :]))
            path_payoff[path] = contract.payoff(fixing_schedule)
        maturity = contract.get_expiry()
        fv = mean(path_payoff) * self._model.get_df(maturity)
        return fv

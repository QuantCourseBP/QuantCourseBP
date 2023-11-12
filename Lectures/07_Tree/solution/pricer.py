from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import math
import numpy as np
from scipy.stats import norm
from src.enums import *
from src.contract import *
from src.model import *
from src.numerical_method import *


class Pricer(ABC):
    # Only for theta calculation via bump and revaluation
    valuation_time: float = 0.0
    relative_bump_size: float = 0.01

    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: Params | MCParams | PDEParams | TreeParams = params

    @staticmethod
    def get_pricers() -> dict[str, Pricer]:
        return {cls.__name__: cls for cls in Pricer.__subclasses__()}

    @classmethod
    def create_pricer(cls, contract: Contract, model: MarketModel, params: Params) -> Pricer:
        instance = cls.__new__(cls)
        instance.__init__(contract, model, params)
        return instance

    @abstractmethod
    def calc_fair_value(self) -> float:
        pass

    def calc_delta(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.relative_bump_size * self.model.spot
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self.model)
            model.bump_spot(b)
            bumped_pricer = self.create_pricer(self.contract, model, self.params)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - bumped_fair_values[1]) / (2 * bump_size)

    def calc_gamma(self, method: GreekMethod) -> float:
        if method != GreekMethod.BUMP:
            self.raise_unsupported_greek_method_error(method, (GreekMethod.BUMP,))
        bump_size = self.relative_bump_size * self.model.spot
        bumped_fair_values = list()
        for b in (bump_size, -bump_size):
            model = copy.deepcopy(self.model)
            model.bump_spot(b)
            bumped_pricer = self.create_pricer(self.contract, model, self.params)
            bumped_fair_values.append(bumped_pricer.calc_fair_value())
            del bumped_pricer
            del model
        return (bumped_fair_values[0] - 2 * self.calc_fair_value() + bumped_fair_values[1]) / (bump_size ** 2)

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
        return 1 / (vol * np.sqrt(time_to_expiry)) * (np.log(spot_over_strike) + (rate + vol ** 2 / 2) * time_to_expiry)

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


class TreePricer(Pricer):
    def __init__(self, contract: Contract, model: MarketModel, params: TreeParams):
        # if not isinstance(contract, EuropeanContract):
        #     raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        if not isinstance(params, TreeParams):
            raise TypeError(f'Params must be of type TreeParams but received {type(params).__name__}')
        super().__init__(contract, model, params)
        if np.isnan(self.params.up_step_mult) or np.isnan(self.params.down_step_mult):
            tree_method = BalancedSimpleBinomialTree(self.contract, self.model, self.params)
        else:
            tree_method = SimpleBinomialTree(self.contract, self.model, self.params)
        self.tree_method = tree_method

    def pre_final_value(self, spot: dict[float, float], step: int, discounted_continuation_value: float) -> float:
        raise RuntimeError('pre_final_value must be overriden in child classes.')

    def calc_fair_value(self) -> float:
        self.tree_method.init_tree()
        spot_tree = self.tree_method.spot_tree
        continuation_value_tree = [[np.nan for _ in level] for level in spot_tree]
        for i in range(len(spot_tree[-1])):
            log_spot = spot_tree[-1][i]
            spot = {self.contract.get_timeline()[0]: np.exp(log_spot)}
            discounted_continuation_value = self.tree_method.df[-1] * self.contract.payoff(spot)
            continuation_value_tree[-1][i] = discounted_continuation_value
        for step in range(self.params.nr_steps - 1, -1, -1):
            for i in range(len(spot_tree[step])):
                log_spot = spot_tree[step][i]
                spot = {self.contract.get_timeline()[0]: np.exp(log_spot)}
                discounted_continuation_value = self.tree_method.prob[1] * continuation_value_tree[step + 1][i] + \
                                                self.tree_method.prob[0] * continuation_value_tree[step + 1][i + 1]
                continuation_value_tree[step][i] = (
                    self.pre_final_value(spot, step, discounted_continuation_value))
        return continuation_value_tree[0][0]


class EuropeanTreePricer(TreePricer):
    def __init__(self, contract: EuropeanContract, model: MarketModel, params: TreeParams):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def pre_final_value(self, spot: dict[float, float], step: int, discounted_continuation_value: float) -> float:
        return discounted_continuation_value


class AmericanTreePricer(TreePricer):
    def __init__(self, contract: AmericanContract, model: MarketModel, params: TreeParams):
        if not isinstance(contract, AmericanContract):
            raise TypeError(f'Contract must be of type AmericanContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def pre_final_value(self, spot: dict[float, float], step: int, discounted_continuation_value: float) -> float:
        intrinsic_value = self.tree_method.df[step] * self.contract.payoff(spot)
        return max(discounted_continuation_value, intrinsic_value) if self.contract.long_short == LongShort.LONG \
            else min(discounted_continuation_value, intrinsic_value)


class EuropeanPDEPricer(Pricer):
    def __init__(self, contract: EuropeanContract, model: MarketModel, params: PDEParams):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        if not isinstance(params, PDEParams):
            raise TypeError(f'Params must be of type PDEParams but received {type(params).__name__}')
        super().__init__(contract, model, params)
        self.contract: EuropeanContract = contract
        self.bs_pde: BlackScholesPDE = BlackScholesPDE(self.contract, self.model, self.params)

    def calc_fair_value(self) -> float:
        if self.params.method == BSPDEMethod.EXPLICIT:
            self.bs_pde.explicit_method()
        elif self.params.method == BSPDEMethod.IMPLICIT:
            self.bs_pde.implicit_method()
        elif self.params.method == BSPDEMethod.CRANK_NICOLSON:
            self.bs_pde.crank_nicolson_method()
        else:
            raise ValueError("Invalid method. Use 'explicit', 'implicit', or 'crank_nicolson'.")

        # linear interpolation
        down = int(np.floor((self.model.spot - self.bs_pde.stock_min) / self.params.und_step))
        up = int(np.ceil((self.model.spot - self.bs_pde.stock_min) / self.params.und_step))
        if down == up:
            return self.bs_pde.grid[down, 0]
        else:
            return self.bs_pde.grid[down, 0] + (self.bs_pde.grid[up, 0] - self.bs_pde.grid[down, 0]) * \
                (self.model.spot - self.bs_pde.stock_min - down * self.params.und_step) / self.params.und_step


class GenericMCPricer(Pricer):
    def __init__(self, contract: Contract, model: MarketModel, params: MCParams):
        super().__init__(contract, model, params)
        if isinstance(model, (FlatVolModel, BSVolModel)):
            self._mc_method = MCMethodFlatVol(self.contract, self.model, self.params)
        else:
            raise TypeError(f'MC is not supported for model type {type(model).__name__}')

    def calc_fair_value_with_ci(self) -> tuple[float, tuple[float, ...]]:
        contract = self.contract
        contractual_timeline = contract.get_timeline()
        spot_paths = self._mc_method.simulate_spot_paths()
        num_of_paths = self.params.num_of_paths
        path_payoff = np.empty(num_of_paths)
        for path in range(num_of_paths):
            fixing_schedule = dict(zip(contractual_timeline, spot_paths[path, :]))
            path_payoff[path] = contract.payoff(fixing_schedule)
        maturity = contract.expiry
        if self.params.control_variate:
            # adjust path_payoff inplace
            self.apply_control_var_adj(path_payoff, spot_paths)
        fv = mean(path_payoff) * self.model.calc_df(maturity)
        fv_conf_interval = tuple([(mean(path_payoff) + 1.96 * mult * np.std(path_payoff, ddof=1) /
                                   np.sqrt(self.params.num_of_paths)) * self.model.calc_df(maturity)
                                  for mult in [-1, 1]])
        return fv, fv_conf_interval

    def calc_fair_value(self) -> float:
        return self.calc_fair_value_with_ci()[0]

    def apply_control_var_adj(self, path_payoff, spot_paths) -> None:
        pricer_cv = self.get_controlvar_helper_pricer(self.contract)
        contract_cv = pricer_cv.contract
        num_of_path = len(path_payoff)
        path_payoff_cv = np.empty(num_of_path)
        for path in range(num_of_path):
            # TODO: pick simulated spots only for the dates which are relevant for the control var contract's payoff
            fixing_schedule = dict(zip(contract_cv.get_timeline(), spot_paths[path, :]))
            path_payoff_cv[path] = contract_cv.payoff(fixing_schedule)
        cov = np.cov(path_payoff, path_payoff_cv)
        b = cov[0][1] / cov[1][1]
        contract_cv_mean = pricer_cv.calc_fair_value() / self.model.calc_df(contract_cv.expiry)
        for i in range(num_of_path):
            path_payoff[i] = path_payoff[i] - b * (path_payoff_cv[i] - contract_cv_mean)

    def get_controlvar_helper_pricer(self, contract: Contract) -> Pricer:
        if isinstance(contract, EuropeanContract):
            und = contract.underlying
            exp = contract.expiry
            contract_cv = ForwardContract(und, LongShort.LONG, 1., exp)
            return ForwardAnalyticPricer(contract_cv, self.model, Params())
        else:
            raise TypeError(f'Control variate is not supported for contract type{type(contract).__name__}')


class AsianMomentMatchingPricer(Pricer):
    def __init__(self, contract: AsianContract, model: MarketModel, params: Params):
        if not isinstance(contract, AsianContract):
            raise TypeError(f'Contract must be of type AsianContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)

    def calc_fair_value(self) -> float:
        direction = self.contract.direction
        strike = self.contract.strike
        expiry = self.contract.expiry
        timeline = self.contract.get_timeline()
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.get_vol(strike, expiry)
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)

        n = len(timeline)
        moment_first = spot / n * sum([np.exp(rate * t_i) for t_i in timeline])
        moment_second = spot ** 2 / n ** 2 * sum(
            [np.exp(rate * (t_i + t_j) + vol ** 2 * min(t_i, t_j)) for t_i in timeline for t_j in timeline])
        param1 = 2 * np.log(moment_first / spot) - np.log(moment_second / spot ** 2) / 2
        param2 = math.sqrt(np.log(moment_second / spot ** 2) - 2 * np.log(moment_first / spot))
        d1 = (np.log(spot / strike) + param1 + param2 ** 2) / param2
        d2 = d1 - param2
        if self.contract.derivative_type == PutCallFwd.CALL:
            return direction * df * (spot * np.exp(param1 + (param2 ** 2) / 2) * norm.cdf(d1) - strike * norm.cdf(d2))
        elif self.contract.derivative_type == PutCallFwd.PUT:
            return direction * df * (
                        strike * norm.cdf(-d2) - (spot * np.exp(param1 + (param2 ** 2) / 2)) * norm.cdf(-d1))
        else:
            self.contract.raise_incorrect_derivative_type_error()


class BarrierAnalyticPricer(Pricer):
    def __init__(self, contract: EuropeanBarrierContract, model: MarketModel, params: Params):
        if not isinstance(contract, EuropeanBarrierContract):
            raise TypeError(f'Contract must be of type EuropeanBarrierContract but received {type(contract).__name__}')
        super().__init__(contract, model, params)
        self._contract: EuropeanBarrierContract = contract

    def calc_fair_value(self) -> float:
        direction = self._contract.direction
        strike = self._contract.strike
        expiry = self._contract.expiry
        time_to_expiry = expiry - self.valuation_time
        spot = self.model.spot
        vol = self.model.get_vol(strike, expiry)
        rate = self.model.risk_free_rate
        df = self.model.calc_df(time_to_expiry)
        barrier = self._contract.barrier.barrier_level
        updown = self._contract.barrier.up_down
        inout = self._contract.barrier.in_out

        if (self._contract.derivative_type == PutCallFwd.CALL) & (updown == UpDown.DOWN) & (inout == InOut.IN):
            part1 = spot * (barrier / spot) ** (2 * rate / vol ** 2 + 1) * \
                    norm.cdf(EuropeanAnalyticPricer.calc_d1(barrier ** 2 / (strike * spot), vol, rate, time_to_expiry))
            part2 = strike * (barrier / spot) ** (2 * rate / vol ** 2 - 1) * \
                    norm.cdf(EuropeanAnalyticPricer.calc_d2(barrier ** 2 / (strike * spot), vol, rate, time_to_expiry))
            return direction * (part1 - df * part2)
        else:
            self.raise_pricer_not_implemented_error()

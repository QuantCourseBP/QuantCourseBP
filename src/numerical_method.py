from __future__ import annotations
from abc import ABC
from src.model import *
import numpy as np
from src.contract import Contract


class NumericalMethod(ABC):
    def __init__(self, model: MarketModel, params: Params) -> None:
        self._model = model
        self._params = params

    @staticmethod
    def get_numerical_methods() -> dict[str, NumericalMethod]:
        return {cls.__name__: cls for cls in NumericalMethod.__subclasses__()}


# todo: to be implemented
class MCMethod(NumericalMethod):
    def __init__(self, model: MarketModel, params: MCParams):
        if not isinstance(params, MCParams):
            raise TypeError('Params must be an instance of class MCParams')
        super().__init__(model, params)

    def find_simulation_tenors(self, contract_timeline: list[float]) -> list[float]:
        final_tenor = max(contract_timeline)
        dt = 1 / self._params.tenor_frequency
        num_of_tenors = int(final_tenor / dt)
        model_tenors = [i * dt for i in range(num_of_tenors)]
        all_simul_tenors = sorted(model_tenors + contract_timeline)
        return all_simul_tenors

    def generate_std_norm(self, num_of_tenors: int) -> np.array:
        np.random.seed(self._params.seed)
        if self._params.antithetic:
            rnd1 = np.random.standard_normal(size=(int(self._params.num_of_paths / 2), num_of_tenors))
            rnd2 = -rnd1
            rnd = np.concatenate((rnd1, rnd2), axis=0)
            if self._params.num_of_paths % 2 == 1:
                zeros = np.zeros((1, self._params.num_of_tenors))
                rnd = np.concatenate((rnd, zeros), axis=0)
        else:
            rnd = np.random.standard_normal(size=(self._params.num_of_paths, self._params.num_of_tenors))
        if self._params.standardize:
            mean = np.mean(rnd)
            std = np.std(rnd)
            rnd = (rnd - mean) / std
        return rnd

    def simulate_spot_paths(self, contract: Contract):
        model = self._model
        contract_tenors = contract.get_timeline()
        simulation_tenors = self.find_simulation_tenors(contract_tenors)
        num_of_tenors = len(simulation_tenors)
        num_of_paths = self._params.num_of_paths
        rnd_num = self.generate_std_norm(num_of_tenors)
        spot_paths = np.empty(shape=(num_of_paths, num_of_tenors))
        initial_spot = model.get_initial_spot()
        vol = model.get_vol(contract.get_strike(), contract.get_expiry())
        for path in range(num_of_paths):
            for t_idx in range(num_of_tenors):
                t_from = simulation_tenors[t_idx - 1]
                t_to = simulation_tenors[t_idx]
                spot_from = initial_spot if t_idx == 0 else spot_paths[path, t_idx - 1]
                z = rnd_num[path, t_idx]
                spot_paths[path, t_idx] = model.evolve_simulated_spot(vol, t_from, t_to, spot_from, z)
        contract_tenor_idx = [idx for idx in range(num_of_tenors) if simulation_tenors[idx] in contract_tenors]
        return spot_paths[:, contract_tenor_idx]


# todo: to be implemented
class PDEMethod(NumericalMethod):
    def __init__(self, model: MarketModel, params: PDEParams):
        if not isinstance(params, PDEParams):
            raise TypeError('Params must be an instance of class PDEParams')
        super().__init__(model, params)


class SimpleBinomialTree(NumericalMethod):
    def __init__(self, params: TreeParams, model: FlatVolModel):
        super().__init__(model, params)
        self._spot_tree_built = False
        self._df_computed = False
        self._prob_computed = False

    def init_tree(self):
        self.build_spot_tree()
        self.compute_df()
        self.compute_prob()

    def build_spot_tree(self):
        if self._spot_tree_built:
            pass
        self._down_log_step = np.log(self._params.down_step_mult)
        self._up_log_step = np.log(self._params.up_step_mult)
        tree = []
        initial_log_spot = np.log(self._model.get_initial_spot())
        previous_level = [initial_log_spot]
        tree += [previous_level]
        for _ in range(self._params.nr_steps):
            new_level = [s + self._down_log_step for s in previous_level]
            new_level += [previous_level[-1] + self._up_log_step]
            tree += [new_level]
            previous_level = new_level

        self._spot_tree = tree
        self._spot_tree_built = True

    def compute_df(self):
        if self._df_computed:
            pass
        delta_t = self._params.exp / self._params.nr_steps
        df_1_step = self._model.get_df(delta_t)
        self._df = [df_1_step ** k for k in range(self._params.nr_steps + 1)]
        self._df_computed = True

    def compute_prob(self):
        if self._prob_computed:
            pass
        if not self._df_computed:
            self._compute_df()
        p = (1 / self._df[1] - np.exp(self._down_log_step)) / (np.exp(self._up_log_step) - np.exp(self._down_log_step))
        q = 1 - p
        self._prob = (p, q)
        self._prob_computed = True


class BalancedSimpleBinomialTree(SimpleBinomialTree):
    def __init__(self, params: TreeParams, model: MarketModel):
        up = BalancedSimpleBinomialTree.calc_up_step_mult(
            model.get_rate(),
            model.get_vol(params.strike, params.exp),
            params.nr_steps,
            params.exp)
        down = BalancedSimpleBinomialTree.calc_down_step_mult(
            model.get_rate(),
            model.get_vol(params.strike, params.exp),
            params.nr_steps,
            params.exp)
        p = TreeParams(params.exp, params.strike, params.nr_steps, up, down)
        super().__init__(p, model)

    @staticmethod
    def calc_up_step_mult(rate: float, vol: float, nr_steps: int, exp: float) -> float:
        delta_t = exp / nr_steps
        log_mean = rate * delta_t - 0.5 * vol ** 2 * delta_t
        return np.exp(log_mean + vol * np.sqrt(delta_t))

    @staticmethod
    def calc_down_step_mult(rate: float, vol: float, nr_steps: int, exp: float) -> float:
        delta_t = exp / nr_steps
        log_mean = rate * delta_t - 0.5 * vol ** 2 * delta_t
        return np.exp(log_mean - vol * np.sqrt(delta_t))


class AnalyticMethod(NumericalMethod):
    def __init__(self, model: MarketModel):
        super().__init__(model, Params())


class Params(ABC):
    def to_dict(self) -> dict[str, object]:
        return vars(self)


class MCParams(Params):
    def __init__(self) -> None:
        # todo: to be implemented, a few examples:
        self.seed: int = 0
        self.num_of_paths: int = 100
        self.timestep: int = 10


class PDEParams(Params):
    def __init__(self, exp: float, strike: float, dtype: PutCallFwd, s_step: int, t_step: int, S_min: int, S_max: int, method: str) -> None:
        # todo: to be implemented
        self.exp = exp  # Time to maturity
        self.s_step = s_step  # dS
        self.t_step = t_step  # dt
        self.strike = strike
        self.contract_type = dtype   # CALL or PUT
        self.S_min = S_min
        self.S_max = S_max
        self.method = method


class TreeParams(Params):
    def __init__(self, exp: float, strike: float, nr_steps: int = 1, up_step_mult: float = np.nan,
                 down_step_mult: float = np.nan) -> None:
        self.exp = exp
        self.nr_steps = nr_steps
        self.up_step_mult = up_step_mult
        self.down_step_mult = down_step_mult
        self.strike = strike


class BlackScholesPDE(PDEMethod):
    def __init__(self, model: MarketModel, params: PDEParams):
        super().__init__(model, params)
        self.exp = params.exp
        self.strike = params.strike
        self.sigma = model.get_vol(self.exp, model.get_initial_spot()/self.strike)
        self.t_step = params.t_step
        self.und_step = params.s_step
        self._derivative_type = params.contract_type
        self.S_min = params.S_min
        self.S_max = params.S_max
        self.ns_steps = int(np.round((self.S_max - self.S_min) / float(self.und_step)))  # Number of time steps
        self.nt_steps = int(np.round(params.exp / float(self.t_step)))  # Number of stock price steps
        self._interest_rate = model.get_rate()
        self.grid = np.zeros((self.nt_steps + 1, self.ns_steps + 1))
        self.P, self.Q, self.R = self.tridiagonal_matrix()

    def setup_boundary_conditions(self):
        if self._derivative_type == PutCallFwd.CALL:
            self.grid[0, :] = np.maximum(np.linspace(self.S_min, self.S_max, self.ns_steps + 1) - self.strike, 0)
            self.grid[:, -1] = (self.S_max - self.strike) * np.exp(
                -self._interest_rate * self.t_step * (self.nt_steps - np.arange(self.nt_steps + 1)))

        else:
            self.grid[0, :] = np.maximum(self.strike - np.linspace(self.S_min, self.S_max, self.ns_steps + 1), 0)
            self.grid[:, -1] = (self.strike - self.S_max) * np.exp(
                -self._interest_rate * self.t_step * (self.nt_steps - np.arange(self.nt_steps + 1)))

    def explicit_method(self):
        self.setup_boundary_conditions()
        for j in range(1, self.nt_steps + 1):
            for i in range(1, self.ns_steps):
                alpha = 0.5 * self.t_step * (self.sigma ** 2 * i ** 2 - self._interest_rate * i)
                beta = 1 - self.t_step * (self.sigma ** 2 * i ** 2 + self._interest_rate)
                gamma = 0.5 * self.t_step * (self.sigma ** 2 * i ** 2 + self._interest_rate * i)
                self.grid[j, i] = alpha * self.grid[j - 1, i - 1] + beta * self.grid[j - 1, i] + gamma * self.grid[
                    j - 1, i + 1]

    def implicit_method(self):
        self.setup_boundary_conditions()
        for j in range(self.nt_steps, 0, -1):
            self.grid[j - 1, :] = np.linalg.solve(self.P, np.dot(self.Q, self.grid[j, :]))

    def crank_nicolson_method(self):
        self.setup_boundary_conditions()
        for j in range(self.nt_steps, 0, -1):
            self.grid[j - 1, :] = np.linalg.solve(self.P, np.dot(self.R, self.grid[j, :]))

    def tridiagonal_matrix(self):
        alpha = -0.5 * self.t_step * (self.sigma ** 2 * np.arange(1, self.ns_steps) ** 2 - self._interest_rate * np.arange(1, self.ns_steps))
        beta = 1 + self.t_step * (self.sigma ** 2 * np.arange(1, self.ns_steps) ** 2 + self._interest_rate)
        gamma = -0.5 * self.t_step * (self.sigma ** 2 * np.arange(1, self.ns_steps) ** 2 + self._interest_rate * np.arange(1, self.ns_steps))

        P = np.diag(alpha[1:], -1) + np.diag(beta) + np.diag(gamma[:-1], 1)
        Q = np.diag(-alpha[1:], -1) + np.diag(1 - beta) + np.diag(-gamma[:-1], 1)
        R = np.diag(alpha[1:], -1) + np.diag(1 - beta) + np.diag(gamma[:-1], 1)

        return P, Q, R



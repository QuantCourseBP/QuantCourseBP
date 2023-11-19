from __future__ import annotations
from abc import ABC
from src.model import *
import numpy as np
from src.contract import Contract, EuropeanContract, AmericanContract


class NumericalMethod(ABC):
    @abstractmethod
    def __init__(self, contract: Contract, model: MarketModel, params: MCParams | PDEParams | TreeParams) -> None:
        self.contract: Contract = contract
        self.model: MarketModel = model
        self.params: MCParams | PDEParams | TreeParams = params

    @staticmethod
    def get_numerical_methods() -> dict[str, NumericalMethod]:
        return {cls.__name__: cls for cls in NumericalMethod.__subclasses__()}


class MCMethod(NumericalMethod):
    def __init__(self, contract: Contract, model: MarketModel, params: MCParams) -> None:
        if not isinstance(params, MCParams):
            raise TypeError(f'Params must be of type MCParams but received {type(params).__name__}')
        super().__init__(contract, model, params)

    def find_simulation_tenors(self) -> list[float]:
        if self.params.tenor_frequency == 0:
            model_tenors = [.0]
        else:
            final_tenor = max(self.contract.get_timeline())
            dt = 1 / self.params.tenor_frequency
            num_of_tenors = int(final_tenor / dt)
            model_tenors = [i * dt for i in range(num_of_tenors)]
        return sorted(set(model_tenors + self.contract.get_timeline()))

    def generate_std_norm(self, num_of_tenors: int) -> np.ndarray:
        np.random.seed(self.params.seed)
        if self.params.antithetic:
            rnd1 = np.random.standard_normal(size=(int(self.params.num_of_paths / 2), num_of_tenors))
            rnd2 = -rnd1
            rnd = np.concatenate((rnd1, rnd2), axis=0)
            if self.params.num_of_paths % 2 == 1:
                zeros = np.zeros((1, num_of_tenors))
                rnd = np.concatenate((rnd, zeros), axis=0)
        else:
            rnd = np.random.standard_normal(size=(self.params.num_of_paths, num_of_tenors))
        if self.params.standardize:
            mean = np.mean(rnd)
            std = np.std(rnd, ddof=1)
            rnd = (rnd - mean) / std
        return rnd

    def simulate_spot_paths(self) -> np.ndarray:
        tenors = self.find_simulation_tenors()
        num_of_paths = self.params.num_of_paths
        num_of_tenors = len(tenors)
        spot_paths = np.zeros(shape=(num_of_paths, num_of_tenors))
        rnd = self.generate_std_norm(num_of_tenors)
        s_0 = self.model.spot
        contract_tenors = self.contract.get_timeline()

        for path in range(num_of_paths):
            for time_index in range(num_of_tenors):
                if time_index == 0:
                    spot_paths[path, time_index] = s_0
                else:
                    t_from = tenors[time_index-1]
                    t_to = tenors[time_index]
                    spot_from = spot_paths[path, time_index-1]
                    z=rnd[path,time_index-1]
                    spot_to = self.evolve_simulated_spot(t_from, t_to, spot_from, z)
                    spot_paths[path, time_index] = spot_to
        contract_tenor_idx = [idx for idx in range(num_of_tenors) if tenors[idx] in contract_tenors]
        return spot_paths[:, contract_tenor_idx]

    @abstractmethod
    def evolve_simulated_spot(self, t_from: float, t_to: float, spot_from: float, z: float) -> float:
        pass


class MCMethodFlatVol(MCMethod):
    def __int__(self, contract: Contract, model: FlatVolModel, params: MCParams):
        super().__init__(contract, model, params)

    def evolve_simulated_spot(self, t_from: float, t_to: float, spot_from: float, z: float) -> float:
        vol = self.model.get_vol(self.contract.strike, self.contract.expiry)
        rate = self.model.risk_free_rate
        dt = t_to - t_from
        if self.params.evolve_spot_method == MCNumMethod.EXACT:
            new_spot = spot_from * np.exp((rate - 0.5 * vol**2) * dt + (vol * z * np.sqrt(dt)))
        elif self.params.evolve_spot_method == MCNumMethod.EULER:
            new_spot = spot_from + rate*spot_from*dt + vol*spot_from*z*np.sqrt(dt)
        else:
            raise TypeError(self.params.evolve_spot_method + " evolve method is not implemented")
        return new_spot


class MCMethodBS(MCMethod):
    def __int__(self, contract: Contract, model: BSVolModel, params: MCParams):
        super().__init__(contract, model, params)

    def evolve_simulated_spot(self, t_from: float, t_to: float, spot_from: float, z: float) -> float:
        vol = self.model.get_vol(self.contract.strike, self.contract.expiry)
        rate = self.model.risk_free_rate
        dt = t_to - t_from
        if self.params.evolve_spot_method == MCNumMethod.EXACT:
            new_spot = spot_from * np.exp((rate - 0.5 * vol**2) * dt + (vol * z * np.sqrt(dt)))
        elif self.params.evolve_spot_method == MCNumMethod.EULER:
            new_spot = spot_from + rate*spot_from*dt + vol*spot_from*z*np.sqrt(dt)
        else:
            raise TypeError(self.params.evolve_spot_method + " evolve method is not implemented")
        return new_spot


class BlackScholesPDE(NumericalMethod):
    def __init__(self, contract: Contract, model: MarketModel, params: PDEParams) -> None:
        if not isinstance(params, PDEParams):
            raise TypeError(f'Params must be of type PDEParams but received {type(params).__name__}')
        if isinstance(contract, AmericanContract):
            self.is_american = True
        elif isinstance(contract, EuropeanContract):
            self.is_american = False
        else:
            self.contract.raise_incorrect_derivative_type_error()

        super().__init__(contract, model, params)
        self.contract = contract
        self.sigma: float = self.model.get_vol(self.contract.strike, self.contract.expiry)
        self.stock_min: float = self.params.stock_min_mult * self.model.spot
        self.stock_max: float = self.params.stock_max_mult * self.model.spot
        # Number of stock price steps
        self.num_of_und_steps: int = int(np.round((self.stock_max - self.stock_min) / float(self.params.und_step)))
        # Number of time steps
        self.num_of_time_steps: int = int(np.round(self.contract.expiry / float(self.params.time_step)))
        self.grid: np.ndarray = np.zeros((self.num_of_und_steps + 1, self.num_of_time_steps + 1))
        self.stock_disc: np.ndarray = np.linspace(self.stock_min, self.stock_max, self.num_of_und_steps + 1)
        self.time_disc: np.ndarray = np.linspace(0, self.contract.expiry, self.num_of_time_steps + 1)
        self.measure_of_stock: np.ndarray = self.stock_disc / self.params.und_step
        self.setup_boundary_conditions()

    def setup_boundary_conditions(self) -> None:
        df = self.model.calc_df(self.contract.expiry - self.time_disc)
        if self.contract.derivative_type == PutCallFwd.CALL:
            # terminal condition
            self.grid[:, -1] = np.maximum(self.stock_disc - self.contract.strike, 0)
            # right boundary
            self.grid[-1, :] = self.stock_max - self.contract.strike * df
        elif self.contract.derivative_type == PutCallFwd.PUT:
            # terminal condition
            self.grid[:, -1] = np.maximum(self.contract.strike - self.stock_disc, 0)
            # left condition
            self.grid[0, :] = self.contract.strike * df - self.stock_min
        else:
            self.contract.raise_incorrect_derivative_type_error()

    def grid_intrinsic_value(self):
        intrinsic_value = None
        if self.is_american:
            if self.contract.derivative_type == PutCallFwd.CALL:
                intrinsic_value = np.maximum(self.stock_disc - self.contract.strike, 0)
            elif self.contract.derivative_type == PutCallFwd.PUT:
                intrinsic_value = np.maximum(self.contract.strike - self.stock_disc, 0)
            else:
                self.contract.raise_incorrect_derivative_type_error()
        return intrinsic_value

    def explicit_method(self) -> None:
        self.setup_boundary_conditions()
        alpha = 0.5 * self.params.time_step * (
                self.sigma ** 2 * self.measure_of_stock ** 2 - self.model.risk_free_rate * self.measure_of_stock)
        beta = 1 - self.params.time_step * (
                self.sigma ** 2 * self.measure_of_stock ** 2 + self.model.risk_free_rate)
        gamma = 0.5 * self.params.time_step * (
                self.sigma ** 2 * self.measure_of_stock ** 2 + self.model.risk_free_rate * self.measure_of_stock)
        for j in range(self.num_of_time_steps-1, -1, -1):  # for t
            for i in range(1, self.num_of_und_steps):  # for S
                self.grid[i, j] = alpha[i] * self.grid[i - 1, j + 1] + beta[i] * self.grid[i, j + 1] + gamma[i] \
                                                  * self.grid[i + 1, j + 1]
                if self.is_american:  # for American Contract
                    intrinsic_value = self.grid_intrinsic_value()[i]
                    self.grid[i, j] = max(self.grid[i, j], intrinsic_value) \
                        if self.contract.long_short == LongShort.LONG else min(self.grid[i, j], intrinsic_value)

    def implicit_method(self) -> None:
        self.setup_boundary_conditions()
        alpha = 0.5 * self.params.time_step * (
                self.model.risk_free_rate * self.measure_of_stock - self.sigma ** 2 * self.measure_of_stock ** 2)
        beta = 1 + self.params.time_step * (
                self.sigma ** 2 * self.measure_of_stock ** 2 + self.model.risk_free_rate)
        gamma = - 0.5 * self.params.time_step * (
                self.sigma ** 2 * self.measure_of_stock ** 2 + self.model.risk_free_rate * self.measure_of_stock)
        upper_matrix = np.diag(alpha[2:-1], -1) + np.diag(beta[1:-1]) + np.diag(gamma[1:-2], 1)
        lower_matrix = np.eye(self.num_of_und_steps - 1)

        rhs_vector = np.zeros(self.num_of_und_steps-1)
        for j in range(self.num_of_time_steps-1, -1, -1):  # for t
            rhs_vector[0] = -alpha[1] * self.grid[0, j+1]
            rhs_vector[-1] = -gamma[-2] * self.grid[-1, j+1]
            self.grid[1:-1, j] = np.linalg.solve(
                lower_matrix, np.linalg.solve(upper_matrix, self.grid[1:-1, j + 1] + rhs_vector))
            if self.is_american:  # for American Contract
                intrinsic_value = self.grid_intrinsic_value()
                self.grid[:, j] = max(self.grid[:, j], intrinsic_value) if self.contract.long_short == LongShort.LONG \
                    else min(self.grid[:, j], intrinsic_value)

    def crank_nicolson_method(self) -> None:
        self.setup_boundary_conditions()
        alpha = 0.25 * self.params.time_step * (
                -self.model.risk_free_rate * self.measure_of_stock + self.sigma ** 2 * self.measure_of_stock ** 2)
        beta = -0.5 * self.params.time_step * (
                self.sigma ** 2 * self.measure_of_stock ** 2 + self.model.risk_free_rate)
        gamma = 0.25 * self.params.time_step * (
                self.sigma ** 2 * self.measure_of_stock ** 2 + self.model.risk_free_rate * self.measure_of_stock)
        upper_matrix = -np.diag(alpha[2:-1], -1) + np.diag(1-beta[1:-1]) - np.diag(gamma[1:-2], 1)
        lower_matrix = np.eye(self.num_of_und_steps - 1)
        rhs_matrix = np.diag(alpha[2:-1], -1) + np.diag(1+beta[1:-1]) + np.diag(gamma[1:-2], 1)

        rhs_vector = np.zeros(self.num_of_und_steps - 1)
        for j in range(self.num_of_time_steps-1, -1, -1):  # for t
            rhs_vector[0] = alpha[1] * (self.grid[0, j + 1] + self.grid[0, j])
            rhs_vector[-1] = gamma[-2] * (self.grid[-1, j + 1] + self.grid[-1, j])
            self.grid[1:-1, j] = np.linalg.solve(
                lower_matrix, np.linalg.solve(upper_matrix, (rhs_matrix @ self.grid[1:-1, j + 1]) + rhs_vector))
            if self.is_american:  # for American Contract
                intrinsic_value = self.grid_intrinsic_value()
                self.grid[:, j] = max(self.grid[:, j], intrinsic_value) if self.contract.long_short == LongShort.LONG \
                    else min(self.grid[:, j], intrinsic_value)


class SimpleBinomialTree(NumericalMethod):
    def __init__(self, contract: Contract, model: MarketModel, params: Params) -> None:
        if not isinstance(params, TreeParams):
            raise TypeError(f'Params must be of type TreeParams but received {type(params).__name__}')
        super().__init__(contract, model, params)
        self.spot_tree_built: bool = False
        self.df_computed: bool = False
        self.prob_computed: bool = False
        self.down_log_step = np.log(self.params.down_step_mult)
        self.up_log_step = np.log(self.params.up_step_mult)
        self.spot_tree: list[list[float]] = list()
        self.df: list[float] = list()
        self.prob: tuple[float, float] = tuple()

    def init_tree(self) -> None:
        self.build_spot_tree()
        self.compute_df()
        self.compute_prob()

    def build_spot_tree(self) -> None:
        if self.spot_tree_built:
            return
        log_spot = np.log(self.model.spot)
        previous_level = [log_spot]
        tree = [previous_level]
        for _ in range(self.params.nr_steps):
            new_level = [s + self.down_log_step for s in previous_level]
            new_level += [previous_level[-1] + self.up_log_step]
            tree += [new_level]
            previous_level = new_level
        self.spot_tree = tree
        self.spot_tree_built = True

    def compute_df(self) -> None:
        if self.df_computed:
            return
        delta_t = self.contract.expiry / self.params.nr_steps
        df_1_step = self.model.calc_df(delta_t)
        self.df = [df_1_step ** k for k in range(self.params.nr_steps + 1)]
        self.df_computed = True

    def compute_prob(self) -> None:
        if self.prob_computed:
            return
        if not self.df_computed:
            self.compute_df()
        prob_up = ((1 / self.df[1] - np.exp(self.down_log_step)) /
                   (np.exp(self.up_log_step) - np.exp(self.down_log_step)))
        self.prob = (prob_up, 1-prob_up)
        self.prob_computed = True


class BalancedSimpleBinomialTree(SimpleBinomialTree):
    def __init__(self, contract: Contract, model: MarketModel, params: Params):
        if not isinstance(params, TreeParams):
            raise TypeError(f'Params must be of type TreeParams but received {type(params).__name__}')
        vol = model.get_vol(contract.strike, contract.expiry) if np.isnan(params.vol) else params.vol
        up = BalancedSimpleBinomialTree.calc_step_mult(
            model.risk_free_rate,
            vol,
            params.nr_steps,
            contract.expiry,
            True)
        down = BalancedSimpleBinomialTree.calc_step_mult(
            model.risk_free_rate,
            vol,
            params.nr_steps,
            contract.expiry,
            False)
        super().__init__(contract, model, TreeParams(params.nr_steps, np.nan, up, down))

    @staticmethod
    def calc_step_mult(rate: float, vol: float, nr_steps: int, exp: float, is_up_direction: bool) -> float:
        direction = 1.0 if is_up_direction else -1.0
        delta_t = exp / nr_steps
        log_mean = rate * delta_t - 0.5 * vol ** 2 * delta_t
        return np.exp(log_mean + direction * vol * np.sqrt(delta_t))


class Params(ABC):
    def to_dict(self) -> dict[str, any]:
        return vars(self)


class MCParams(Params):
    def __init__(self, seed: int = 1, num_of_path: int = 10000, tenor_frequency: int = 4,
                 standardize: bool = True, antithetic: bool = True, control_variate: bool = False,
                 evolve_spot_method: MCNumMethod = MCNumMethod.EULER) -> None:
        self.seed = seed
        self.num_of_paths = num_of_path
        self.tenor_frequency = tenor_frequency
        self.standardize = standardize
        self.antithetic = antithetic
        self.control_variate = control_variate
        self.evolve_spot_method = evolve_spot_method


class PDEParams(Params):
    def __init__(self, und_step: int = 2, time_step: float = 1/1200, stock_min_mult: float = 0,
                 stock_max_mult: float = 2, method: BSPDEMethod = BSPDEMethod.EXPLICIT) -> None:
        self.und_step = und_step  # dS
        self.time_step = time_step  # dt
        self.stock_min_mult = stock_min_mult
        self.stock_max_mult = stock_max_mult
        self.method = method


class TreeParams(Params):
    def __init__(self, nr_steps: int = 1, vol: float = np.nan, up_step_mult: float = np.nan,
                 down_step_mult: float = np.nan) -> None:
        self.nr_steps = nr_steps
        self.up_step_mult = up_step_mult
        self.down_step_mult = down_step_mult
        self.vol = vol

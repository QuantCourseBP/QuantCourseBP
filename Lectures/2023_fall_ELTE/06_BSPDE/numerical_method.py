from __future__ import annotations
from abc import ABC
from src.model import *
import numpy as np
from src.contract import Contract, EuropeanContract
import scipy
from src.numerical_method import MCParams, TreeParams


# TASK:
# Implement the setup_boundary_conditions method for BlackScholesPDE
# This should include terminal and boundary conditions based on a Call/Put option


class NumericalMethod(ABC):
    @abstractmethod
    def __init__(self, contract: Contract, model: MarketModel, params: MCParams | PDEParams | TreeParams):
        self._contract = contract
        self._model = model
        self._params = params

    @staticmethod
    def get_numerical_methods() -> dict[str, NumericalMethod]:
        return {cls.__name__: cls for cls in NumericalMethod.__subclasses__()}


class PDEMethod(NumericalMethod):
    def __init__(self, contract: Contract, model: MarketModel, params: Params):
        if not isinstance(params, PDEParams):
            raise TypeError(f'Params must be of type PDEParams but received {type(params).__name__}')
        super().__init__(contract, model, params)


class Params(ABC):
    def to_dict(self) -> dict[str, any]:
        return vars(self)


class PDEParams(Params):
    def __init__(self, und_step: int = 2, time_step: float = 1/1200, stock_min_mult: float = 0, stock_max_mult: float = 2,
                 method: BSPDEMethod = BSPDEMethod.EXPLICIT) -> None:
        self.und_step = und_step  # dS
        self.time_step = time_step  # dt
        self.stock_min_mult = stock_min_mult
        self.stock_max_mult = stock_max_mult
        self.method = method


class BlackScholesPDE(PDEMethod):
    def __init__(self, contract: EuropeanContract, model: MarketModel, params: PDEParams):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        if not isinstance(params, PDEParams):
            raise TypeError(f'Params must be of type PDEParams but received {type(params).__name__}')
        super().__init__(contract, model, params)
        self.contract = contract
        self.exp = contract.expiry
        self.strike = contract.strike
        self.sigma = model.get_vol(contract.strike, contract.expiry)
        self.time_step = params.time_step
        self.und_step = params.und_step
        self.derivative_type = contract.derivative_type
        self.stock_min = params.stock_min_mult * model.spot
        self.stock_max = params.stock_max_mult * model.spot
        self.num_of_und_steps = int(np.round((self.stock_max - self.stock_min) / float(self.und_step)))  # Number of stock price steps
        self.num_of_time_steps = int(np.round(self.exp / float(self.time_step)))   # Number of time steps
        self.interest_rate = model.risk_free_rate
        self.grid = np.zeros((self.num_of_und_steps + 1, self.num_of_time_steps + 1))
        self.stock_disc = np.linspace(self.stock_min, self.stock_max, self.num_of_und_steps + 1)
        self.time_disc = np.linspace(0, self.exp, self.num_of_time_steps + 1)
        self.measure_of_stock = self.stock_disc / self.und_step
        self.df = model.calc_df(self.exp - self.time_disc)

    def setup_boundary_conditions(self):
        pass
        # if self.derivative_type == PutCallFwd.CALL:
        #     # terminal condition
        #     # self.grid[:, -1]
        #     # right boundary
        #     # self.grid[-1, :]
        #
        # elif self.derivative_type == PutCallFwd.PUT:
        #     # terminal condition
        #     self.grid[:, -1] =
        #     # left condition
        #     self.grid[0, :] =
        #
        # else:
        #     self.contract.raise_incorrect_derivative_type_error()

    def explicit_method(self):
        self.setup_boundary_conditions()
        alpha = 0.5 * self.time_step * (self.sigma ** 2 * self.measure_of_stock ** 2 - self.interest_rate *
                                        self.measure_of_stock)
        beta = 1 - self.time_step * (self.sigma ** 2 * self.measure_of_stock ** 2 + self.interest_rate)
        gamma = 0.5 * self.time_step * (self.sigma ** 2 * self.measure_of_stock ** 2 + self.interest_rate *
                                        self.measure_of_stock)
        for j in range(self.num_of_time_steps-1, -1, -1):  # for t
            for i in range(1, self.num_of_und_steps):  # for S
                self.grid[i, j] = alpha[i] * self.grid[i - 1, j + 1] + beta[i] * self.grid[i, j + 1] + gamma[i] \
                                              * self.grid[i + 1, j + 1]

    def implicit_method(self):
        self.setup_boundary_conditions()
        alpha = 0.5 * self.time_step * (self.interest_rate * self.measure_of_stock - self.sigma ** 2 *
                                        self.measure_of_stock ** 2)
        beta = 1 + self.time_step * (self.sigma ** 2 * self.measure_of_stock ** 2 + self.interest_rate)
        gamma = - 0.5 * self.time_step * (self.sigma ** 2 * self.measure_of_stock ** 2 +
                                          self.interest_rate * self.measure_of_stock)
        upper_matrix = np.diag(alpha[2:-1], -1) + np.diag(beta[1:-1]) + np.diag(gamma[1:-2], 1)
        lower_matrix = np.eye(self.num_of_und_steps - 1)

        rhs_vector = np.zeros(self.num_of_und_steps-1)
        for j in range(self.num_of_time_steps-1, -1, -1):  # for t
            rhs_vector[0] = -alpha[1] * self.grid[0, j+1]
            rhs_vector[-1] = -gamma[-2] * self.grid[-1, j+1]
            self.grid[1:-1, j] = np.linalg.solve(lower_matrix, np.linalg.solve(upper_matrix, self.grid[1:-1, j + 1]
                                                                               + rhs_vector))

    def crank_nicolson_method(self):
        self.setup_boundary_conditions()

        alpha = 0.25 * self.time_step * (-self.interest_rate * self.measure_of_stock + self.sigma ** 2 *
                                         self.measure_of_stock ** 2)
        beta = -0.5 * self.time_step * (self.sigma ** 2 * self.measure_of_stock ** 2 + self.interest_rate)
        gamma = 0.25 * self.time_step * (self.sigma ** 2 * self.measure_of_stock ** 2 +
                                         self.interest_rate * self.measure_of_stock)
        upper_matrix = - np.diag(alpha[2:-1], -1) + np.diag(1-beta[1:-1]) - np.diag(gamma[1:-2], 1)
        lower_matrix = np.eye(self.num_of_und_steps - 1)
        rhs_matrix = np.diag(alpha[2:-1], -1) + np.diag(1+beta[1:-1]) + np.diag(gamma[1:-2], 1)

        rhs_vector = np.zeros(self.num_of_und_steps - 1)
        for j in range(self.num_of_time_steps-1, -1, -1):  # for t
            rhs_vector[0] = alpha[1] * (self.grid[0, j + 1] + self.grid[0, j])
            rhs_vector[-1] = gamma[-2] * (self.grid[-1, j + 1] + self.grid[-1, j])
            self.grid[1:-1, j] = np.linalg.solve(lower_matrix,
                                                 np.linalg.solve(upper_matrix,
                                                                 (rhs_matrix @ self.grid[1:-1, j + 1]) + rhs_vector))




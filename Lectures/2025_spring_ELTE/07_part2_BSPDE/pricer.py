from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from src.numerical_method import *


# TASK:
# Implement the calc_fair_value method for the EuropeanPDEPricer based on the different numerical method


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
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.spot
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
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.spot
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
        strike = self._contract.strike
        expiry = self._contract.expiry
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
        bump_size = self.RELATIVE_BUMP_SIZE * self._model.risk_free_rate
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
            supported: tuple[GreekMethod, ...] = (_.value for _ in GreekMethod)) -> None:
        raise ValueError(f'Unsupported GreekMethod {method} for Pricer {type(self).__name__}. '
                         f'Supported methods are: {", ".join(supported)}')


class EuropeanPDEPricer(Pricer):
    def __init__(self, contract: EuropeanContract, model: MarketModel, params: PDEParams):
        if not isinstance(contract, EuropeanContract):
            raise TypeError(f'Contract must be of type EuropeanContract but received {type(contract).__name__}')
        if not isinstance(params, PDEParams):
            raise TypeError(f'Params must be of type PDEParams but received {type(params).__name__}')
        self._derivative_type = contract.derivative_type
        self._bsPDE = BlackScholesPDE(contract, model, params)
        self.grid = self._bsPDE.grid
        self._initial_spot = model.spot
        self.und_step = params.und_step
        self.time_step = params.time_step
        self.stock_min = self._bsPDE.stock_min
        self.stock_max = self._bsPDE.stock_max
        self.method = params.method

    def calc_fair_value(self) -> float:
        pass
        # if self.method == BSPDEMethod.EXPLICIT:
        #
        # elif self.method == BSPDEMethod.IMPLICIT:
        #
        # elif self.method == BSPDEMethod.CRANK_NICOLSON:
        #
        # else:
        #     raise ValueError("Invalid method. Use 'explicit', 'implicit', or 'crank_nicolson'.")

        # linear interpolation
        down = int(np.floor((self._initial_spot - self.stock_min)/self.und_step))
        up = int(np.ceil((self._initial_spot - self.stock_min)/self.und_step))

        if down == up:
            return self.grid[down, 0]
        else:
            return self.grid[down, 0] + (self.grid[up, 0] - self.grid[down, 0]) * \
                   (self._initial_spot - self.stock_min - down*self.und_step)/self.und_step

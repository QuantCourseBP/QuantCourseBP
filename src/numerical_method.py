from __future__ import annotations
from abc import ABC
from src.model import *
import numpy as np


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

    def generate_std_norm(self) -> np.array:

        np.random.seed(self._params.seed)

        if self._params.antithetic:
            rnd1 = np.random.standard_normal(size=(int(self._params.num_of_paths / 2), self._params.num_of_tenors))
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


class MCMethodFlatVol(MCMethod):
    def __init__(self, model: FlatVolModel, params: MCParams):
        super().__init__(model, params)

    def create_spot_paths(self) -> np.array:

        std_norm = self.generate_std_norm()

        rf_rate = self._model.get_rate()
        vol = self._model.vol
        log_proc_drift = rf_rate - 1 / 2 * vol * vol

        s0 = self.index.level

        log_ret_paths = np.zeros((self.num_of_paths, self.num_of_tenors + 1))

        for path in range(self.num_of_paths):
            log_ret_paths[path][0] = 0

        for path in range(self.num_of_paths):
            for tenor_idx in range(1, self.num_of_tenors + 1):
                d_time = self.tenors[tenor_idx] - self.tenors[tenor_idx - 1]
                log_ret_paths[path][tenor_idx] = log_ret_paths[path][
                                                     tenor_idx - 1] + log_proc_drift * d_time + vol * np.sqrt(d_time) * \
                                                 std_norm[path][tenor_idx - 1]
        return s0 * np.exp(log_ret_paths)

# todo: to be implemented
class PDEMethod(NumericalMethod):
    def __init__(self, model: MarketModel, params: PDEParams):
        if not isinstance(params, PDEParams):
            raise TypeError('Params must be an instance of class PDEParams')
        super().__init__(model, params)


# todo: to be implemented
class TreeMethod(NumericalMethod):
    def __init__(self, model: MarketModel, params: TreeParams):
        if not isinstance(params, TreeParams):
            raise TypeError('Params must be an instance of class TreeParams')
        super().__init__(model, params)


class AnalyticMethod(NumericalMethod):
    def __new__():
        raise NotImplementedError('AnalyticMethod class cannot be instantiated')


class Params(ABC):
    def to_dict(self) -> dict[str, object]:
        return vars(self)


class MCParams(Params):
    def __init__(self) -> None:
        # todo: to be implemented, a few examples:
        self.seed: int = 0
        self.num_of_paths: int = 100
        self.tenors: np.array = [1]
        self.standardize: bool = True
        self.antithetic: bool = True


class PDEParams(Params):
    def __init__(self) -> None:
        # todo: to be implemented
        pass


class TreeParams(Params):
    def __init__(self) -> None:
        # todo: to be implemented
        pass

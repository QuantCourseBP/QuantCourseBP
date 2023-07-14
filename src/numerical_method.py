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
        if(self._spot_tree_built):
            pass
        self._down_log_step = np.log(self._params._down_step_mult)
        self._up_log_step = np.log(self._params._up_step_mult)
        tree = []
        initial_log_spot = np.log(1.0)#np.log(self.model.spot)
        previous_level = [initial_log_spot]
        tree += [previous_level]
        for _ in range(self._params._nr_steps):
            new_level = [s + self._down_log_step for s in previous_level]
            new_level += [previous_level[-1] + self._up_log_step]
            tree += [new_level]
            previous_level = new_level
        
        self._spot_tree = tree
        self._spot_tree_built = True
        
    def compute_df(self):
        if(self._df_computed):
            pass
        delta_t = self._params._exp / self._params._nr_steps
        df_1_step = self._model.get_df(delta_t)
        self._df = [df_1_step**k for k in range(self._params._nr_steps + 1)]
        self._df_computed = True
        
    def compute_prob(self):
        if(self._prob_computed):
            pass
        if(not self._df_computed):
            self._compute_df()
        p = (1/self._df[1] - np.exp(self._down_log_step))/(np.exp(self._up_log_step) - np.exp(self._down_log_step))
        q = 1 - p
        self._prob = (p, q)
        self._prob_computed = True
    
class BalancedSimpleBinomialTree(SimpleBinomialTree):
    def __init__(self, params: TreeParams, model: FlatVolModel):
        super().__init__(TreeParams(params._exp, params._moneyness, params._nr_steps, BalancedSimpleBinomialTree.calc_up_step_mult(model.get_rate(), model.get_vol(params._exp, params._moneyness), params._nr_steps, params._exp), BalancedSimpleBinomialTree.calc_down_step_mult(model.get_rate(), model.get_vol(params._exp, params._moneyness), params._nr_steps, params._exp)), model)
        
    @staticmethod
    def calc_up_step_mult(rate: float, vol: float, nr_steps: int, exp: float):
        rate = rate
        delta_t = exp / nr_steps
        log_mean = rate * delta_t - 0.5 * vol**2 * delta_t
        return np.exp(log_mean + vol * np.sqrt(delta_t))
        
    @staticmethod
    def calc_down_step_mult(rate: float, vol: float, nr_steps: int, exp: float):
        rate = rate
        delta_t = exp / nr_steps
        log_mean = rate * delta_t - 0.5 * vol**2 * delta_t
        return np.exp(log_mean - vol * np.sqrt(delta_t))


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
        self.timestep: int = 10


class PDEParams(Params):
    def __init__(self) -> None:
        # todo: to be implemented
        pass


class TreeParams(Params):
    def __init__(self, exp: float, moneyness: float, nr_steps: int = 1, up_step_mult: float = np.nan, down_step_mult: float = np.nan) -> None:
        self._exp = exp
        self._nr_steps = nr_steps
        self._up_step_mult = up_step_mult
        self._down_step_mult = down_step_mult
        self._moneyness = moneyness

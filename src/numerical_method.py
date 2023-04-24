from __future__ import annotations
from abc import ABC


class NumericalMethod(ABC):
    @staticmethod
    def get_numerical_methods() -> dict[str, NumericalMethod]:
        return {cls.__name__: cls for cls in NumericalMethod.__subclasses__()}


# todo: to be implemented
class MonteCarloMethod(NumericalMethod):
    pass


# todo: to be implemented
class PDEMethod(NumericalMethod):
    pass


# todo: to be implemented
class TreeMethod(NumericalMethod):
    pass


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
    def __init__(self) -> None:
        # todo: to be implemented
        pass

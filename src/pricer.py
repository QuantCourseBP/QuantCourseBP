from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *
from src.contract import *
from src.model import *
from src.numerical_method import *


class Pricer(ABC):
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

    @abstractmethod
    def calc_delta(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_gamma(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_vega(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_theta(self, method: GreekMethod) -> float:
        pass

    @abstractmethod
    def calc_rho(self, method: GreekMethod) -> float:
        pass


# todo: to be implemented
class EuropeanAnalyticPricer(Pricer):
    pass


# todo: to be implemented
class GenericTreePricer(Pricer):
    pass


# todo: to be implemented
class GenericPDEPricer(Pricer):
    pass


# todo: to be implemented
class GenericMCPricer(Pricer):
    pass

from __future__ import annotations
from abc import ABC, abstractmethod
from src.enums import *


class MarketModel(ABC):
    def __init__(self, und: Stock):
        self._und = und

    @staticmethod
    def get_models() -> dict[str, MarketModel]:
        return {cls.__name__: cls for cls in MarketModel.__subclasses__()}


# todo: to be implemented
class BSVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)


# todo: to be implemented
class FlatVolModel(MarketModel):
    def __init__(self, und: Stock):
        super().__init__(und)

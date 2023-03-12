from __future__ import annotations
from enums import *


class MarketData:
    # todo: change this to the proper path
    __MARKET_FOLDER: str = './path/to/folder/'
    __initial_spot: dict[Stock, float] = dict()
    __volgrid: dict[Stock, Volgrid] = dict()

    def __new__(cls) -> None:
        raise TypeError('Static class cannot be instantiated')

    @staticmethod
    def __load_initial_spot() -> None:
        # todo: to be implemented
        pass

    @staticmethod
    def __load_volgrid() -> None:
        # todo: to be implemented
        pass

    @staticmethod
    def initialize() -> None:
        MarketData.__load_initial_spot()
        MarketData.__load_volgrid()

    @staticmethod
    def get_initial_spot() -> dict[Stock, float]:
        return MarketData.__initial_spot

    @staticmethod
    def get_volgrid() -> dict[Stock, Volgrid]:
        return MarketData.__volgrid


# todo: needs to be implemented, defined temporarily as empty class due to annotations
class Volgrid:
    pass

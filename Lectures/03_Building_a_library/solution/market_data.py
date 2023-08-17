from __future__ import annotations
from src.enums import *


class MarketData:
    __risk_free_rate: float = 0.05
    __spot: dict[Stock, float] = {
        Stock.BLUECHIP_BANK: 130.17,
        Stock.TIPTOP_SOLUTIONS: 177.32
    }
    __volatility: dict[Stock, float] = {
        Stock.BLUECHIP_BANK: 0.381,
        Stock.TIPTOP_SOLUTIONS: 0.320
    }

    def __new__(cls) -> None:
        raise RuntimeError('Static class cannot be instantiated')

    @staticmethod
    def get_risk_free_rate() -> float:
        return MarketData.__risk_free_rate

    @staticmethod
    def get_spot() -> dict[Stock, float]:
        return MarketData.__spot

    @staticmethod
    def get_vol() -> dict[Stock, float]:
        return MarketData.__volatility

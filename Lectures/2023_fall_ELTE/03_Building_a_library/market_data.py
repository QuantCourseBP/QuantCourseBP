from __future__ import annotations
from src.enums import *


# TASK:
# 1. Define the necessary market data (spot, volatility, risk-free rate) in MarketData class.
#    Risk-free rate: 5%
#    Underlying                |   Spot    |   Volatility
#    Stock.BLUECHIP_BANK       |   130.17  |   0.381
#    Stock.TIPTOP_SOLUTIONS    |   177.32  |   0.320


class MarketData:
    risk_free_rate = None
    spot = None
    volatility: dict[Stock, float] = {
        Stock.BLUECHIP_BANK: 0.381,
        Stock.TIPTOP_SOLUTIONS: 0.320
    }

    def __new__(cls) -> None:
        raise RuntimeError('Static class cannot be instantiated')

    @staticmethod
    def get_risk_free_rate() -> float:
        return MarketData.risk_free_rate

    @staticmethod
    def get_spot() -> dict[Stock, float]:
        return MarketData.spot

    @staticmethod
    def get_vol() -> dict[Stock, float]:
        return MarketData.volatility

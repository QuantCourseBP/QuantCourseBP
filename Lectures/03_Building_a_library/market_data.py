from __future__ import annotations
from src.enums import *

# TASK:
# Define the necessary market data (spot, volatility, risk-free rate) in MarketData class.
# Risk-free rate: 5%
# Underlying                |   Spot    |   Volatility
# Stock.BLUECHIP_BANK       |   130.17  |   0.381
# Stock.TIPTOP_SOLUTIONS    |   177.32  |   0.320


class MarketData:
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

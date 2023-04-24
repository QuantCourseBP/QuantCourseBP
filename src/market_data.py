from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from src.enums import *


class MarketData:
    __MARKET_FOLDER: str = os.path.join(os.path.dirname(__file__), '..', 'mkt')
    __FILENAME_INIT_SPOT: str = 'spot.csv'
    __FILENAME_VOL_GRID: str = 'vol_{und}.csv'
    __initial_spot: dict[Stock, float] = dict()
    __vol_grid: dict[Stock, VolGrid] = dict()
    __is_initialized: bool = False

    def __new__(cls) -> None:
        raise TypeError('Static class cannot be instantiated')

    @staticmethod
    def __load_initial_spot() -> None:
        path = os.path.join(MarketData.__MARKET_FOLDER, MarketData.__FILENAME_INIT_SPOT)
        data = pd.read_csv(path, header=0, index_col=0, dtype={'underlying': str, 'initial_spot': float})
        for stock in Stock:
            name = stock.value
            if name not in data.index:
                raise ValueError(f'Missing initial spot for underlying: {name}')
            MarketData.__initial_spot[stock] = data.at[name, 'initial_spot']

    @staticmethod
    def __load_vol_grid() -> None:
        for stock in Stock:
            name = stock.value
            filename = MarketData.__FILENAME_VOL_GRID.format(und=name.lower())
            path = os.path.join(MarketData.__MARKET_FOLDER, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f'Volatility grid ({filename}) does not exist for underlying: {name}')
            data = pd.read_csv(path, header=0, dtype=float)
            points = np.array((data['tenor'], data['moneyness'])).T
            values = np.array(data['value'])
            MarketData.__vol_grid[stock] = VolGrid(stock, points, values)

    @staticmethod
    def initialize() -> None:
        MarketData.__load_initial_spot()
        MarketData.__load_vol_grid()
        MarketData.__is_initialized = True

    @staticmethod
    def __validate() -> None:
        if not MarketData.__is_initialized:
            raise ValueError(
                'Market data is not initialized. Call MarketData.initialize() in the first line after imports.')

    @staticmethod
    def get_initial_spot() -> dict[Stock, float]:
        MarketData.__validate()
        return MarketData.__initial_spot

    @staticmethod
    def get_vol_grid() -> dict[Stock, VolGrid]:
        MarketData.__validate()
        return MarketData.__vol_grid


class VolGrid:
    def __init__(self, und: Stock, points: np.ndarray, values: np.ndarray) -> None:
        self.__und: Stock = und
        self.__points: np.ndarray = points
        self.__values: np.ndarray = values
        if not (self.__values.ndim == 1 and self.__points.ndim == 2 and self.__points.shape[1] == 2
                and self.__points.shape[0] == self.__values.shape[0]):
            raise AssertionError('Incorrect dimensions for volatility grid points and values')
        self.__interpolator = LinearInterpolatorNearestExtrapolator(self.__points, self.__values)

    def get_und(self) -> Stock:
        return self.__und

    def get_points(self) -> np.ndarray:
        return self.__points

    def get_values(self) -> np.ndarray:
        return self.__values

    def get_vol(self, tenor: float, moneyness: float) -> float:
        return self.__interpolator((tenor, moneyness))


class LinearInterpolatorNearestExtrapolator:
    def __init__(self, points: np.ndarray, values: np.ndarray) -> None:
        self.__func_linear = LinearNDInterpolator(points, values)
        self.__func_nearest = NearestNDInterpolator(points, values)

    def __call__(self, *args) -> float:
        t = self.__func_linear(*args)
        if np.isnan(t):
            return self.__func_nearest(*args)
        return t.item(0)

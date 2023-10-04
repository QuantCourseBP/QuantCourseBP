from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from enums import *


class MarketData:
    market_folder: str = os.path.join(os.path.dirname(__file__), '..', '..', 'mkt')
    filename_spot: str = 'spot.csv'
    filename_vol_grid: str = 'vol_{und}.csv'
    risk_free_rate: float = 0.05
    spot: dict[Stock, float] = dict()
    volgrid: dict[Stock, VolGrid] = dict()
    is_initialized: bool = False

    def __new__(cls) -> None:
        raise RuntimeError('Static class cannot be instantiated')

    @staticmethod
    def load_spot() -> None:
        path = os.path.join(MarketData.market_folder, MarketData.filename_spot)
        data = pd.read_csv(path, header=0, index_col=0, dtype={'underlying': str, 'spot': float})
        for stock in Stock:
            name = stock.value
            if name not in data.index:
                raise ValueError(f'Missing spot for underlying: {name}')
            MarketData.spot[stock] = data.at[name, 'spot']

    @staticmethod
    def load_volgrid() -> None:
        for stock in Stock:
            name = stock.value
            filename = MarketData.filename_vol_grid.format(und=name.lower())
            path = os.path.join(MarketData.market_folder, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f'Volatility grid ({filename}) does not exist for underlying: {name}')
            data = pd.read_csv(path, header=0, dtype=float)
            points = np.array((data['strike'], data['expiry']), dtype='float64').T
            values = np.array(data['volatility'], dtype='float64')
            MarketData.volgrid[stock] = VolGrid(stock, points, values)

    @staticmethod
    def initialize() -> None:
        MarketData.load_spot()
        MarketData.load_volgrid()
        MarketData.is_initialized = True

    @staticmethod
    def validate() -> None:
        if not MarketData.is_initialized:
            raise ValueError(
                'Market data is not initialized. Call MarketData.initialize() in the first line after imports.')

    @staticmethod
    def get_risk_free_rate() -> float:
        MarketData.validate()
        return MarketData.risk_free_rate

    @staticmethod
    def get_spot() -> dict[Stock, float]:
        MarketData.validate()
        return MarketData.spot

    @staticmethod
    def get_volgrid() -> dict[Stock, VolGrid]:
        MarketData.validate()
        return MarketData.volgrid


class VolGrid:
    def __init__(self, underlying: Stock, points: np.ndarray, values: np.ndarray) -> None:
        self.underlying: Stock = underlying
        self.points: np.ndarray = points
        self.values: np.ndarray = values
        if not (self.values.ndim == 1 and self.points.ndim == 2 and self.points.shape[1] == 2
                and self.points.shape[0] == self.values.shape[0]):
            raise AssertionError('Incorrect dimensions for volatility grid points and values')
        self.interpolator = LinearInterpolatorNearestExtrapolator(self.points, self.values)

    def get_vol(self, strike_expiry_pairs: np.ndarray) -> np.ndarray:
        """
        Interpolates/extrapolates volatility at vector of volgrid coordinates.
        :param strike_expiry_pairs: Array of (strike, expiry) coordinates to interpolate at.
        :return: Array of interpolated/extrapolated vols.
        """
        return self.interpolator(strike_expiry_pairs)


class LinearInterpolatorNearestExtrapolator:
    def __init__(self, points: np.ndarray, values: np.ndarray) -> None:
        self.func_linear = LinearNDInterpolator(points, values)
        self.func_nearest = NearestNDInterpolator(points, values)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate interpolator at given points.
        Piecewise linear interpolation inside the grid, flat extrapolation outside the grid.
        :param points: Array of coordinates to interpolate at.
        :return: Array of interpolated values.
        """
        interpolation = self.func_linear(points)
        mask = np.isnan(interpolation)
        interpolation[mask] = self.func_nearest(points[mask])
        return interpolation

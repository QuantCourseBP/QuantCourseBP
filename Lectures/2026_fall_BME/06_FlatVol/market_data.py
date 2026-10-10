from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from enums import *
from pathlib import Path


class MarketData:
    risk_free_rate = 0.05
    spot: dict[Stock, float] = {
        Stock.BLUECHIP_BANK: 130.17,
        Stock.TIPTOP_SOLUTIONS: 177.32
    }
    volatility: dict[Stock, float] = {
        Stock.BLUECHIP_BANK: 0.381,
        Stock.TIPTOP_SOLUTIONS: 0.320
    }

    def __new__(cls) -> None:
        raise RuntimeError('Static class cannot be instantiated')

    @staticmethod
    def load_spot() -> None:
        raise NotImplementedError

    @staticmethod
    def load_volgrid() -> None:
        raise NotImplementedError

    @staticmethod
    def initialize() -> None:
        raise NotImplementedError

    @staticmethod
    def validate() -> None:
        raise NotImplementedError

    @staticmethod
    def get_risk_free_rate() -> float:
        return MarketData.risk_free_rate

    @staticmethod
    def get_spot() -> dict[Stock, float]:
        return MarketData.spot

    @staticmethod
    def get_vol() -> dict[Stock, float]:
        return MarketData.volatility


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

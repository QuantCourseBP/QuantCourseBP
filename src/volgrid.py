from __future__ import annotations
from enums import *
import pandas as pd
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

class LinearNDInterpolatorExt(object):
    def __init__(self, points,values):
        self.funcinterp=LinearNDInterpolator(points,values)
        self.funcnearest=NearestNDInterpolator(points,values)
    def __call__(self,*args):
        t=self.funcinterp(*args)
        if not np.isnan(t):
            return t.item(0)
        else:
            return self.funcnearest(*args)

class VolGrid:
    def __init__(self, und: Stock):
        self.und = und
        volgrid_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "mkt", f"{und.name}_vol.csv"))
        self.points = np.array((volgrid_data.tenor, volgrid_data.moneyness)).T
        self.values = volgrid_data.value

    def get_und(self):
        return self.und

    def get_vol(self, maturity, moneyness):
        interp =  LinearNDInterpolatorExt( self.points, self.values)
        return interp((maturity,moneyness))


    def get_volgrid_data(self):
        return self.volgrid_data

if __name__ == "__main__":
    vol_grid = VolGrid(Stock.EXAMPLE1)
    print(vol_grid.get_und())
    print(vol_grid.get_vol(0.65, 1))

    vol_grid = VolGrid(Stock.EXAMPLE2)
    print(vol_grid.get_und())
    print(vol_grid.get_vol(3, 4))

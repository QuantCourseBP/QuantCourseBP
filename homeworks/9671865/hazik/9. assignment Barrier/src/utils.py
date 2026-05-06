import numpy as np
from src.market_data import *
from src.enums import *
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import Akima1DInterpolator


def prob_breach_barrier_segment(barrier: float, vol: float, t1: float, t2: float,
                                val1: float, val2: float, up_down: UpDown) -> float:
    if up_down == UpDown.DOWN:
        if barrier > min(val1, val2):
            return 1.0
        else:
            return np.exp(-2 * np.log(val1 / barrier) * np.log(val2 / barrier) / (vol ** 2 * (t2 - t1)))
    else:
        if barrier < max(val1, val2):
            return 1.0
        else:
            return np.exp(-2 * np.log(barrier / val1) * np.log(barrier / val2) / (vol ** 2 * (t2 - t1)))


def plot_vol_surface(volgrid: VolGrid, num_steps=30, show_obs=True, view=(25, 50)) -> None:
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    points = volgrid.points.T
    strike_range = (min(points[0]), max(points[0]))
    expiry_range = (min(points[1]), max(points[1]))
    strike = np.linspace(strike_range[0], strike_range[1], num_steps)
    expiry = np.linspace(expiry_range[0], expiry_range[1], num_steps)
    strike, expiry = np.meshgrid(strike, expiry)
    vols = volgrid.get_vol(np.array([strike, expiry]).reshape(2, num_steps**2).T)
    vols = vols.reshape(num_steps, num_steps)
    ax.plot_surface(strike, expiry, vols, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.8)
    if show_obs:
        strike = points[0]
        expiry = points[1]
        vols = volgrid.values
        ax.scatter3D(strike, expiry, vols, s=10, color='black')
    ax.set_xlabel('Strike', fontweight='bold')
    ax.set_ylabel('Expiry', fontweight='bold')
    ax.set_zlabel('Vol', fontweight='bold')
    ax.view_init(view[0], view[1])
    fig.suptitle('Implied volatility surface', fontweight='bold')
    fig.tight_layout()
    plt.show()


def plot_vol_slice(volgrid: VolGrid, expiry: float, linear_interpolation: bool = False):
    fig, ax = plt.subplots()
    points = volgrid.points.T
    all_strike = points[0]
    all_expiry = points[1]
    all_vol = volgrid.values
    mask = np.isclose(all_expiry, expiry)
    if all(~mask):
        tenors = sorted(list(np.unique(all_expiry)))
        raise ValueError(f'No data for requested tenor. Please choose from: {tenors}')
    strike = all_strike[mask]
    vol = all_vol[mask]
    grid = np.linspace(min(strike), max(strike), 100)
    if linear_interpolation:
        interpolation = np.interp(grid, strike, vol)
    else:
        akima = Akima1DInterpolator(strike, vol)
        interpolation = akima(grid)
    ax.plot(strike, vol, 'o')
    ax.plot(grid, interpolation, '#1f77b4')
    ax.set_xlabel('Strike', fontweight='bold')
    ax.set_ylabel('Vol', fontweight='bold')
    fig.suptitle('Implied volatility', fontweight='bold')
    ax.set_title(f'Expiry: {round(expiry, 2)} years')
    plt.show()

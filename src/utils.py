import numpy as np
from src.market_data import *
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_vol_surface(volgrid: VolGrid, num_steps=30, show_obs=True, view=(25, 50)) -> None:
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    points = volgrid.get_points().T
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
        vols = volgrid.get_values()
        ax.scatter3D(strike, expiry, vols, s=10, color='black')
    ax.set_xlabel('Strike', fontweight='bold')
    ax.set_ylabel('Expiry', fontweight='bold')
    ax.set_zlabel('Vol', fontweight='bold')
    ax.view_init(view[0], view[1])
    plt.tight_layout()
    plt.show()

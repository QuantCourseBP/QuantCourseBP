# Demo B: one-step tree, several contracts (expiries T, strikes K), flat rate (yearly DF = 0.95)
import numpy as np
from scipy.optimize import brentq
from plotter import my_plotter
from hw1_call import black_scholes_eur_call
from bg_functions import balanced_tree_call_price
from calibrator import calibrate, tree_vol

spot, df_annual, sigma = 1.0, 0.95, 0.2
r = -np.log(df_annual)
expiries = [0.25, 0.5, 1.0, 2.0]
strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
u_table, vol_table = [], []
for T in expiries:
    u_row, vol_row = [], []
    for K in strikes:
        price = black_scholes_eur_call(r=r, T=T, S0=spot, sigma=sigma, K=K)[0]
        u = calibrate(price, spot, K, T, 1, df_annual)       # one step: DF = df_annual**T
        DF = df_annual ** T
        u_brentq = brentq(lambda x: balanced_tree_call_price(spot, K, T, 1, df_annual, x)
                          - price, 1 / DF + 1e-10, 2 / DF - 1e-10, xtol=1e-14)
        assert abs(u - u_brentq) < 1e-9                          # another root solver
        u_row.append(u)
        vol_row.append(tree_vol(u, T, 1, df_annual))
    u_table.append(u_row)
    vol_table.append(vol_row)
    print(f"T = {T:4.2f}  u = " + "  ".join(f"{u:.4f}" for u in u_row)
          + "   tree vol = " + "  ".join(f"{v:.3f}" for v in vol_row))

common = {"grid": True, "title_fontsize": 13, "label_fontsize": 11, "legend_fontsize": 10,
          "linewidth": 2, "markers": "o", "hline": None, "show": False}
names = [f"T = {T}" for T in expiries]
my_plotter(strikes, u_table, names=names,
           layout={**common, "title": "Demo B: one-step tree, calibrated up step",
                   "x_label": "strike K", "y_label": "up step u", "legend_loc": "upper center",
                   "save_path": "figures/demoB_u.pdf"})
my_plotter(strikes, vol_table, names=names,
           layout={**common, "title": "Demo B: one-step tree, tree volatility",
                   "x_label": "strike K", "y_label": "tree volatility",
                   "save_path": "figures/demoB_vol.pdf"})

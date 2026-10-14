# Demo A: the same contract (S0 = K = 1, T = 1, yearly DF = 0.95) with several numbers of steps
import numpy as np
from plotter import my_plotter
from hw1_call import black_scholes_eur_call
from bg_functions import balanced_tree_call_price
from calibrator import calibrate, tree_vol

spot, strike, maturity, df_annual, sigma = 1.0, 1.0, 1.0, 0.95, 0.2
r = -np.log(df_annual)
target = black_scholes_eur_call(r=r, T=maturity, S0=spot, sigma=sigma, K=strike)[0]
steps_list = [1, 2, 3, 5, 10, 20, 50, 100]
u_list, vol_list = [], []
for steps in steps_list:
    DF = df_annual ** (maturity / steps)                     # scaled DF: flat rate (Def. 3)
    u = calibrate(target, spot, strike, maturity, steps, df_annual)
    error = balanced_tree_call_price(spot, strike, maturity, steps, df_annual, u) - target
    u_list.append(u)
    vol_list.append(tree_vol(u, maturity, steps, df_annual))
    print(f"N = {steps:3d}  DF = {DF:.6f}  u = {u:.6f}  d = {2/DF - u:.6f}  "
          f"tree vol = {vol_list[-1]:.4f}  error = {error:.1e}")

common = {"grid": True, "title_fontsize": 13, "label_fontsize": 11, "legend_fontsize": 10,
          "linewidth": 2, "markers": "o", "logx": True, "show": False}
my_plotter(steps_list, u_list, names=["calibrated u"],
           layout={**common, "title": "Demo A: calibrated up step against N",
                   "x_label": "number of steps N", "y_label": "up step u", "hline": 1.0,
                   "save_path": "figures/demoA_u.pdf"})
my_plotter(steps_list, vol_list, names=["tree volatility (u - 1/DF)/sqrt(dt)"],
           layout={**common, "title": "Demo A: tree volatility against N",
                   "x_label": "number of steps N", "y_label": "tree volatility",
                   "hline": sigma, "save_path": "figures/demoA_vol.pdf"})

# Q1: test of calibrate (N = 1 against another root solver, prices out of range, picture)
import numpy as np
from scipy.optimize import brentq
from plotter import my_plotter
from hw1_call import black_scholes_eur_call
from bg_functions import balanced_tree_call_price
from calibrator import calibrate, price_range

spot, strike, maturity, df_annual = 1.0, 1.0, 1.0, 0.95
r = -np.log(df_annual)                                        # the same flat rate
target = black_scholes_eur_call(r=r, T=maturity, S0=spot, sigma=0.2, K=strike)[0]
print(f"target price (Black-Scholes of Homework 1, sigma = 0.2): {target:.6f}")

# (a) one step: calibrate against scipy.optimize.brentq on V(u) - V* (another root solver)
u = calibrate(target, spot, strike, maturity, 1, df_annual)
DF = df_annual ** maturity
u_brentq = brentq(lambda x: balanced_tree_call_price(spot, strike, maturity, 1, df_annual, x)
                  - target, 1 / DF + 1e-10, 2 / DF - 1e-10, xtol=1e-14)
print(f"N = 1: calibrate u = {u:.10f},  brentq u = {u_brentq:.10f}")

# (b) prices that the tree cannot give (Step 2): calibrate raises ValueError
for price in [0.60, 0.04]:
    try:
        calibrate(price, spot, strike, maturity, 1, df_annual)
    except ValueError as error:
        print("ValueError:", error)

# (c) picture: tree price against u for N = 1, 2, 5; the calibrated u is where it meets the target
u_grid = np.linspace(1.0, 1.6, 121)
curves, names, u_cal = [], [], []
for steps in [1, 2, 5]:
    DF = df_annual ** (maturity / steps)
    curves.append([balanced_tree_call_price(spot, strike, maturity, steps, df_annual, x)
                   if 1 / DF < x < 2 / DF else np.nan for x in u_grid])
    u_cal.append(calibrate(target, spot, strike, maturity, steps, df_annual))
    names.append(f"tree, N = {steps}  (u = {u_cal[-1]:.4f})")
fig, ax = my_plotter(u_grid, curves,
                     layout={"title": "Calibration: tree price against the up step u",
                             "x_label": "up step u", "y_label": "call price",
                             "hline": target, "vlines": u_cal, "grid": True,
                             "title_fontsize": 13, "label_fontsize": 11, "legend_fontsize": 10,
                             "linewidth": 2, "colors": ["tab:blue", "tab:orange", "tab:green"],
                             "save_path": "figures/q1_calibration.pdf", "show": False},
                     names=names)

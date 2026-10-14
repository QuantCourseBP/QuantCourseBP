import json
import numpy as np
from scipy.optimize import brentq
from bg_functions import balanced_tree_call_price
from hw1_call import black_scholes_eur_call
from calibrator import calibrate, price_range, tree_vol

out = {}
spot, strike, maturity, df_annual, sigma = 1.0, 1.0, 1.0, 0.95, 0.2
r = -np.log(df_annual)
target = black_scholes_eur_call(r=r, T=maturity, S0=spot, sigma=sigma, K=strike)[0]
print(f"target price (Black-Scholes, sigma = 0.2): {target:.6f}")
out["target"] = target
# Q1 tests
u1 = calibrate(target, spot, strike, maturity, 1, df_annual)
u1_brentq = brentq(lambda x: balanced_tree_call_price(spot, strike, maturity, 1, df_annual, x)
                   - target, 1 / df_annual + 1e-10, 2 / df_annual - 1e-10, xtol=1e-14)
print(f"N = 1: u = {u1:.10f}, brentq u = {u1_brentq:.10f}")
out["u1"], out["u1_brentq"] = u1, u1_brentq
for bad in [0.60, 0.04]:
    try:
        calibrate(bad, spot, strike, maturity, 1, df_annual)
    except ValueError as e:
        print("ValueError:", e)
# Demo A
rowsA = []
for steps in [1, 2, 3, 5, 10, 20, 50, 100]:
    u = calibrate(target, spot, strike, maturity, steps, df_annual)
    DF = df_annual ** (maturity / steps)
    err = balanced_tree_call_price(spot, strike, maturity, steps, df_annual, u) - target
    rowsA.append(dict(N=steps, DF=DF, u=u, d=2 / DF - u, h=u - 1 / DF,
                      vol=tree_vol(u, maturity, steps, df_annual), err=err))
    print(f"N = {steps:3d}  DF = {DF:.6f}  u = {u:.6f}  d = {2/DF-u:.6f}  "
          f"h = {u-1/DF:.6f}  tree vol = {rowsA[-1]['vol']:.4f}  error = {err:.1e}")
out["demoA"] = rowsA
# Demo B
rowsB = []
for T in [0.25, 0.5, 1.0, 2.0]:
    for K in [0.8, 0.9, 1.0, 1.1, 1.2]:
        V = black_scholes_eur_call(r=r, T=T, S0=spot, sigma=sigma, K=K)[0]
        u = calibrate(V, spot, K, T, 1, df_annual)
        DF = df_annual ** T
        lo, hi = price_range(spot, K, T, 1, df_annual)
        u_b = brentq(lambda x: balanced_tree_call_price(spot, K, T, 1, df_annual, x) - V,
                     1 / DF + 1e-10, 2 / DF - 1e-10, xtol=1e-14)
        rowsB.append(dict(T=T, K=K, V=V, u=u, brentq=u_b,
                          vol=tree_vol(u, T, 1, df_annual), lo=lo, hi=hi))
        print(f"T = {T:4.2f}  K = {K:.1f}  V = {V:.6f}  range = ({lo:.4f}, {hi:.4f})  "
              f"u = {u:.6f}  brentq = {u_b:.6f}  tree vol = {rowsB[-1]['vol']:.4f}")
out["demoB"] = rowsB
# price curves for the picture of the calibration (N = 1, 2, 5)
curves = {}
for steps in [1, 2, 5]:
    DF = df_annual ** (maturity / steps)
    ug = np.linspace(1 / DF + 1e-6, min(2 / DF - 1e-6, 1.6), 120)
    curves[steps] = [[float(u), balanced_tree_call_price(spot, strike, maturity, steps, df_annual, u)] for u in ug]
out["curves"] = curves
json.dump(out, open("results.json", "w"), indent=1)

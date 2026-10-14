# Q4: the calibrated balanced trees of Demo A (S0 = K = 1, T = 1, yearly DF = 0.95), N = 2, 5, 50
import numpy as np
from hw1_call import black_scholes_eur_call
from tree_plot import calibrated_trees, important_nodes, plot_tree

spot, strike, maturity, df_annual = 1.0, 1.0, 1.0, 0.95
r = -np.log(df_annual)
target = black_scholes_eur_call(r=r, T=maturity, S0=spot, sigma=0.2, K=strike)[0]
print(f"target price: {target:.6f}")

for steps in [2, 5, 50]:
    u, d, DF, spot_tree, price_tree = calibrated_trees(target, spot, strike, maturity, steps,
                                                       df_annual)
    print(f"\nN = {steps}: u = {u:.6f}, d = {d:.6f}, DF = {DF:.6f}, "
          f"root value = {price_tree[0][0]:.6f}")
    for t in sorted({0, 1, steps // 2, steps}):
        nodes = [f"(S={spot_tree[t][i]:.4f}, V={price_tree[t][i]:.6f})" for i in important_nodes(t)]
        for k in range(0, len(nodes), 3):                    # at most 3 nodes per line
            print(f"  t = {t:2d}: " + "  ".join(nodes[k:k + 3]))
    # backward induction check at the root: V_0 = (V_up + V_down) / 2 (p_up = p_down = 1/2)
    print(f"  (V_up + V_down)/2 at t = 1: {(price_tree[1][0] + price_tree[1][1]) / 2:.6f}")
    plot_tree(spot_tree, price_tree, strike=strike,
              title=f"Calibrated balanced tree, N = {steps} (u = {u:.4f})")[0].savefig(
        f"figures/tree_N{steps}.pdf")

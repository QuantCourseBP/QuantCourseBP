"""Homework 2, part 2 (Q4): pictures of the calibrated balanced trees.

calibrated_trees(...) calibrates u with calibrate (Q1) and builds the two trees of the lecture
code: the spot tree (create_spot_tree) and the price tree (create_discounted_price_tree).
plot_tree(...) draws them: x = time step t, y = spot S_{t,i}, colour = value of the node as
returned by create_discounted_price_tree (discounted to t = 0; the root is the price).
Small trees: every node is labelled. Large trees: all nodes are drawn, labels only on the
important levels (first, second, middle, last) and on the top, middle and bottom node.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from calibrator import calibrate
from prof_tree import calcBalancedDownStep, create_discounted_price_tree, create_spot_tree


def calibrated_trees(price: float, spot: float, strike: float, maturity: float, steps: int,
                     df_annual: float):
    """Return (u, d, DF, spot_tree, price_tree) of the balanced tree calibrated to price."""
    u = calibrate(price, spot, strike, maturity, steps, df_annual)
    discount_factor = df_annual ** (maturity / steps)              # flat rate (Definition 3)
    d = calcBalancedDownStep(u, discount_factor)                   # lecture, cell 13
    spot_tree = create_spot_tree(spot, u, d, steps)                # lecture, cell 5
    price_tree = create_discounted_price_tree(spot_tree, discount_factor, strike)   # cell 8
    return u, d, discount_factor, spot_tree, price_tree


def important_nodes(level: int) -> list[int]:
    """Indices i of the nodes of level t to label: all of them if few, else bottom/middle/top."""
    return list(range(level + 1)) if level <= 5 else sorted({0, level // 2, level})


def plot_tree(spot_tree, price_tree, strike=None, label_levels=None, title=None, ax=None,
              fontsize=8, logy=None):
    """Draw a spot tree with its price tree. Returns (fig, ax)."""
    steps = len(spot_tree) - 1
    if label_levels is None:
        label_levels = range(steps + 1) if steps <= 5 else sorted({0, steps // 2, steps})
    if logy is None:
        logy = steps > 5                     # log scale: the large tree looks symmetric
    if ax is None:
        ax = plt.subplots(figsize=(9, 5))[1]
    fig = ax.figure
    # edges: node (t, i) -> down child (t+1, i) and up child (t+1, i+1)
    segments = [[(t, spot_tree[t][i]), (t + 1, spot_tree[t + 1][j])]
                for t in range(steps) for i in range(t + 1) for j in (i, i + 1)]
    ax.add_collection(LineCollection(segments, colors="0.75", linewidths=0.6, zorder=1))
    # nodes, coloured by the value of the price tree
    xs = [t for t in range(steps + 1) for _ in spot_tree[t]]
    ys = [s for level in spot_tree for s in level]
    vs = [v for level in price_tree for v in level]
    size = 60 if steps <= 5 else max(4, 400 / steps)
    points = ax.scatter(xs, ys, c=vs, cmap="viridis", s=size, zorder=2)
    fig.colorbar(points, ax=ax, label="node value (discounted to t = 0)")
    for t in label_levels:
        for i in important_nodes(t):
            ax.annotate(f"S={spot_tree[t][i]:.3f}\nV={price_tree[t][i]:.4f}",
                        (t, spot_tree[t][i]), textcoords="offset points", xytext=(4, 4),
                        fontsize=fontsize, zorder=3,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.8, "lw": 0})
    if strike is not None:
        ax.axhline(strike, linestyle="--", color="tab:red", linewidth=1, label=f"strike K = {strike}")
        ax.legend(loc="upper left", fontsize=fontsize + 1)
    if logy:
        ax.set_yscale("log")
        ticks = [2.0 ** k for k in range(-6, 7) if min(ys) <= 2.0 ** k <= max(ys)]
        ax.set_yticks(ticks, [f"{x:g}" for x in ticks])
        ax.minorticks_off()
    ax.set_xlabel("time step t")
    ax.set_ylabel("spot S" + (" (log scale)" if logy else ""))
    ax.set_xlim(-0.3, steps + 0.8 + 0.08 * steps)
    if title:
        ax.set_title(title)
    return fig, ax

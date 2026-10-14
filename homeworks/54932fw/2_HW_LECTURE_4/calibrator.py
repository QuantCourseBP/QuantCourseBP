"""Homework 2, part 2 (Task): calibrator for the Balanced Binomial Tree.

calibrate(...) returns the up step u for which the balanced tree (p_up = p_down = 0.5)
prices a given European call at a given price. Steps 1-5 follow the flowchart of the report.
"""
import numpy as np
from scipy.optimize import minimize

from bg_functions import balanced_tree_call_price


def price_range(spot: float, strike: float, maturity: float, steps: int,
                df_annual: float) -> tuple[float, float]:
    """Tree prices at the two ends of the interval of u (Lemma 1): u_low + eps, u_high - eps.
    The price does not decrease in u (statement of the professor), so every u gives a price
    between these two values."""
    discount_factor = df_annual ** (maturity / steps)
    u_low, u_high = 1 / discount_factor, 2 / discount_factor
    eps = 1e-10 * u_low
    price_min = balanced_tree_call_price(spot, strike, maturity, steps, df_annual, u_low + eps)
    price_max = balanced_tree_call_price(spot, strike, maturity, steps, df_annual, u_high - eps)
    return price_min, price_max


def tree_vol(spot_mult_up: float, maturity: float, steps: int, df_annual: float) -> float:
    """Tree volatility (u - 1/DF) / sqrt(dt) of part 1 (inverse of up_step_from_tree_vol)."""
    dt = maturity / steps
    return (spot_mult_up - 1 / df_annual ** dt) / np.sqrt(dt)


def calibrate(price: float, spot: float, strike: float, maturity: float, steps: int,
              df_annual: float, n_grid: int = 40, n_zoom: int = 3,
              tol: float = 1e-10) -> float:
    """Up step u such that balanced_tree_call_price(..., u) = price."""
    # Step 1: discount factor of one step (flat rate, Definition 3) and the interval of u
    discount_factor = df_annual ** (maturity / steps)
    u_low, u_high = 1 / discount_factor, 2 / discount_factor      # d < 1/DF < u and d > 0
    # Step 2: is the price attainable? (tree prices at the two ends of the interval)
    price_min, price_max = price_range(spot, strike, maturity, steps, df_annual)
    if not price_min < price < price_max:
        raise ValueError(f"price {price:.6f} not attainable: it must lie in "
                         f"({price_min:.6f}, {price_max:.6f})")
    # Step 3: bracket the solution. The price is non-decreasing in u (C2), so on a grid of u
    #         the target lies between the last point with price < target and the first point
    #         with price >= target. Zoom in (new grid inside the bracket): use at least n_zoom
    #         grids, and stop when the left end is out of the flat part (price > price_min;
    #         near 1/DF the price can be constant).
    def tree_price(u: float) -> float:
        return balanced_tree_call_price(spot, strike, maturity, steps, df_annual, u)
    eps = 1e-10 * u_low
    u_left, u_right = u_low + eps, u_high - eps
    for zoom in range(30):
        grid = np.linspace(u_left, u_right, n_grid)
        k = next(i for i, u in enumerate(grid) if tree_price(u) >= price)
        u_left, u_right = grid[max(k - 1, 0)], grid[k]
        if zoom + 1 >= n_zoom and tree_price(u_left) > price_min + 1e-12:
            break
    # Step 4: minimize the squared pricing error inside the bracket (hint of the assignment)
    result = minimize(lambda x: (tree_price(x[0]) - price) ** 2, x0=[(u_left + u_right) / 2],
                      method="Powell", bounds=[(u_left, u_right)],
                      options={"xtol": 1e-14, "ftol": 1e-30})
    spot_mult_up = float(np.atleast_1d(result.x)[0])
    # Step 5: check the result
    error = tree_price(spot_mult_up) - price
    if abs(error) > tol:
        raise RuntimeError(f"calibration failed: pricing error {error:.2e}")
    return spot_mult_up

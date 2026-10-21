"""Calibrator for the balanced binomial tree (Task 2).

Run from the repository root (virtual environment active):
    python homeworks\\29481ec_tree\\tree_calibrator.py
It prints the calibration tables and saves three figures next to this file.
"""
from pathlib import Path


def save_figure(name):
    import matplotlib.pyplot as plt
    out = Path(__file__).resolve().parent / name
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print('Figure saved to', out)

# # Calibrator for the Balanced Binomial Tree
#
# **Task:** for a given number of steps, find the *up step size* `u` for which a **Balanced Binomial Tree**
# (risk-neutral up/down probabilities both 0.5) returns a given market price of a European call.
#
# **Idea**
# * In a balanced tree the down step is determined by the up step: `d = 2/df - u` (`df` = discount factor of one step), so `u` is the only free parameter - it plays the role of the volatility.
# * The tree price is a non-decreasing function of `u`, and it is strictly increasing on the part that matters, so for a given market price there is exactly one `u` (1-1 correspondence).
# * `calibrate` therefore (1) checks that the price is attainable, (2) locates the interval of `u` that contains the solution on a coarse grid, (3) refines it with `scipy.optimize.minimize` (minimising the squared price error inside that interval).


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['axes.grid'] = True


# ## 1. Tree code from the lecture
# (Same functions as in `Binomial_tree_coding.ipynb`, without the diagnostic printing. The only change: the price tree reads the up/down *multipliers* as `spot_tree[1][...] / spot`, so it also works for `spot != 1`.)


def european_call_payoff(S: float, K: float) -> float:
    return max(S - K, 0.0)


def create_spot_tree(spot, spot_mult_up, spot_mult_down, steps):
    previous_level = [spot]
    tree = [previous_level]
    for _ in range(steps):
        new_level = [s * spot_mult_down for s in previous_level]
        new_level += [previous_level[-1] * spot_mult_up]
        tree += [new_level]
        previous_level = new_level
    return tree


def create_discounted_price_tree(spot_tree, discount_factor, K):
    spot = spot_tree[0][0]
    spot_mult_up = spot_tree[1][-1] / spot
    spot_mult_down = spot_tree[1][0] / spot
    p_up = (1 / discount_factor - spot_mult_down) / (spot_mult_up - spot_mult_down)
    p_down = 1 - p_up
    steps = len(spot_tree) - 1
    tree = [[np.nan for _ in level] for level in spot_tree]
    for i in range(len(spot_tree[-1])):                       # leaves
        tree[-1][i] = discount_factor ** steps * european_call_payoff(spot_tree[-1][i], K)
    for step in range(steps - 1, -1, -1):                      # backward induction
        for i in range(len(spot_tree[step])):
            tree[step][i] = p_up * tree[step + 1][i] + p_down * tree[step + 1][i + 1]
    return tree


def calc_balanced_down_step(spot_mult_up, discount_factor):
    return spot_mult_up - 2 * (spot_mult_up - 1 / discount_factor)


# ## 2. Tree price as a function of the up step
# The discount factor is **scaled with the number of steps** (`df = exp(-r*T/steps)`) to keep the interest rate flat.


def step_discount_factor(rate, maturity, steps):
    return np.exp(-rate * maturity / steps)


def tree_call_price(spot, strike, rate, maturity, steps, spot_mult_up):
    df = step_discount_factor(rate, maturity, steps)
    spot_mult_down = calc_balanced_down_step(spot_mult_up, df)
    spot_tree = create_spot_tree(spot, spot_mult_up, spot_mult_down, steps)
    price_tree = create_discounted_price_tree(spot_tree, df, strike)
    return price_tree[0][0]


# ## 3. The calibrator
#
# Admissible up steps: `1/df < u < 2/df`
# * `u > 1/df` keeps the tree arbitrage free (`d < 1/df < u`),
# * `u < 2/df` keeps the down step positive.


def calibrate(target_price, spot, strike, rate, maturity, steps, grid_points=60):
    """Up step size of the balanced binomial tree that reproduces `target_price` of a European call.

    Returns a dict with the calibrated up/down step, the model price and the pricing error.
    """
    df = step_discount_factor(rate, maturity, steps)
    eps = 1e-9
    u_low, u_high = 1 / df + eps, 2 / df - eps               # admissible range of the up step
    price_of = lambda u: tree_call_price(spot, strike, rate, maturity, steps, u)

    # 1) coarse grid: is the target attainable, and where is the solution?
    grid = np.linspace(u_low, u_high, grid_points)
    grid_prices = np.array([price_of(u) for u in grid])
    if not grid_prices[0] < target_price < grid_prices[-1]:
        raise ValueError(f'Price {target_price:.4f} cannot be reached with {steps} step(s): '
                         f'attainable range is ({grid_prices[0]:.4f}, {grid_prices[-1]:.4f})')
    j = np.max(np.where(grid_prices < target_price)[0])
    bracket = (grid[j], grid[j + 1])                          # price crosses the target inside

    # 2) refine with a minimiser inside the bracket
    result = minimize(lambda x: (price_of(x[0]) - target_price) ** 2,
                      x0=[0.5 * (bracket[0] + bracket[1])], bounds=[bracket], method='L-BFGS-B',
                      options={'ftol': 1e-30, 'gtol': 1e-20, 'eps': 1e-10, 'maxiter': 1000})
    u = result.x[0]
    model_price = price_of(u)
    if abs(model_price - target_price) > 1e-6:
        raise RuntimeError('Calibration did not converge')
    return {'up': u, 'down': calc_balanced_down_step(u, df),
            'model_price': model_price, 'error': model_price - target_price}


# ## 4. Market prices
# In practice the option price is observed in the market. To have realistic inputs we generate "market" prices
# with the Black-Scholes formula (volatility 20%, flat rate 5%).


def bs_call(S, K, r, T, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def equivalent_vol(u, rate, maturity, steps):
    """Volatility sigma for which the lecture formula u = exp((r - sigma^2/2) dt + sigma sqrt(dt)) gives this u."""
    dt = maturity / steps
    c = np.log(u) - rate * dt
    return (1 - np.sqrt(1 - 2 * c)) / np.sqrt(dt)


SPOT, RATE, SIGMA = 100.0, 0.05, 0.20


# ## 5. One calibration, step by step
# Call with `K = 105`, `T = 1`, **one-step** tree. The price is a monotone function of `u`, so the target price is hit exactly once.


K, T = 105.0, 1.0
market_price = bs_call(SPOT, K, RATE, T, SIGMA)
print(f'Market price: {market_price:.6f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, steps in zip(axes, [1, 10]):
    df = step_discount_factor(RATE, T, steps)
    us = np.linspace(1 / df + 1e-6, 2 / df - 1e-6, 300)
    ax.plot(us, [tree_call_price(SPOT, K, RATE, T, steps, u) for u in us], label='tree price')
    ax.axhline(market_price, color='gray', linestyle='--', label='market price')
    sol = calibrate(market_price, SPOT, K, RATE, T, steps)
    ax.plot(sol['up'], sol['model_price'], 'ro', label=f"calibrated u = {sol['up']:.4f}")
    ax.set_title(f'{steps} step(s): tree price vs up step')
    ax.set_xlabel('up step size u')
    ax.set_ylabel('price')
    ax.legend()
    print(f"steps={steps:3d}: u={sol['up']:.6f}, d={sol['down']:.6f}, price error={sol['error']:.1e}")
plt.tight_layout()
save_figure('01_price_vs_up_step.png')


# ## 6. Same contract, different number of steps
# The calibrated up step shrinks as the number of steps grows (the time step gets shorter). Expressed as an equivalent volatility it approaches the 20% used to generate the market price.


K, T = 105.0, 1.0
market_price = bs_call(SPOT, K, RATE, T, SIGMA)
steps_list = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
rows = []
print(f'Contract: K={K}, T={T}, market price = {market_price:.6f}\n')
print(f"{'steps':>6} {'up step u':>11} {'down step d':>12} {'model price':>12} {'error':>9} {'equiv. vol':>11}")
for n in steps_list:
    sol = calibrate(market_price, SPOT, K, RATE, T, n)
    vol = equivalent_vol(sol['up'], RATE, T, n)
    rows.append((n, sol['up'], vol))
    print(f"{n:6d} {sol['up']:11.6f} {sol['down']:12.6f} {sol['model_price']:12.6f} {sol['error']:9.1e} {vol:11.4f}")

n_arr, u_arr, vol_arr = (np.array(c) for c in zip(*rows))
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(n_arr, u_arr, 'o-')
axes[0].set_xscale('log')
axes[0].set_title('Calibrated up step vs number of steps')
axes[0].set_xlabel('number of steps')
axes[0].set_ylabel('up step size u')
axes[1].plot(n_arr, vol_arr, 'o-', label='equivalent volatility of the calibrated tree')
axes[1].axhline(SIGMA, color='gray', linestyle='--', label='volatility behind the market price (20%)')
axes[1].set_xscale('log')
axes[1].set_title('Calibrated tree approaches Black-Scholes')
axes[1].set_xlabel('number of steps')
axes[1].set_ylabel('volatility')
axes[1].legend()
plt.tight_layout()
save_figure('02_calibrated_vs_steps.png')


# ## 7. One-step tree, various contracts
# Different expiries and strike levels, flat interest rate. For each contract we observe the market price and calibrate the one-step tree.


expiries = [0.25, 0.5, 1.0, 2.0]
strikes = [90, 95, 100, 105, 110, 115, 120]
up_steps = {}
print(f"{'T':>5} {'K':>5} {'market price':>13} {'up step u':>10} {'down step d':>12} {'error':>9}")
for T in expiries:
    up_steps[T] = []
    for K in strikes:
        price = bs_call(SPOT, K, RATE, T, SIGMA)
        try:
            sol = calibrate(price, SPOT, K, RATE, T, steps=1)
            up_steps[T].append(sol['up'])
            print(f"{T:5.2f} {K:5d} {price:13.6f} {sol['up']:10.6f} {sol['down']:12.6f} {sol['error']:9.1e}")
        except ValueError as e:
            up_steps[T].append(np.nan)
            print(f'{T:5.2f} {K:5d}  not calibratable: {e}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for T in expiries:
    axes[0].plot(strikes, up_steps[T], 'o-', label=f'T = {T}')
axes[0].set_title('Calibrated up step vs strike (one-step tree)')
axes[0].set_xlabel('strike K')
axes[0].set_ylabel('up step size u')
axes[0].legend()
for k, K in enumerate(strikes):
    if K in (90, 100, 110, 120):
        axes[1].plot(expiries, [up_steps[T][k] for T in expiries], 'o-', label=f'K = {K}')
axes[1].set_title('Calibrated up step vs expiry (one-step tree)')
axes[1].set_xlabel('expiry T (years)')
axes[1].set_ylabel('up step size u')
axes[1].legend()
plt.tight_layout()
save_figure('03_one_step_contracts.png')


# ## 8. Sanity checks


# 1) the tree built with the calibrated step reproduces the market price, and its risk-neutral probabilities are 0.5/0.5
K, T, n = 105.0, 1.0, 20
price = bs_call(SPOT, K, RATE, T, SIGMA)
sol = calibrate(price, SPOT, K, RATE, T, n)
df = step_discount_factor(RATE, T, n)
p_up = (1 / df - sol['down']) / (sol['up'] - sol['down'])
assert abs(tree_call_price(SPOT, K, RATE, T, n, sol['up']) - price) < 1e-6
assert abs(p_up - 0.5) < 1e-12
print('risk-neutral up probability:', p_up)

# 2) a higher market price needs a higher step size (higher step = higher volatility)
prices = [bs_call(SPOT, 105.0, RATE, 1.0, s) for s in (0.1, 0.2, 0.3, 0.4)]
us = [calibrate(p, SPOT, 105.0, RATE, 1.0, 10)['up'] for p in prices]
assert all(a < b for a, b in zip(us, us[1:]))
print('up steps for vol 10%, 20%, 30%, 40%:', np.round(us, 4))

# 3) a price that cannot be produced by the tree is rejected
try:
    calibrate(1000.0, SPOT, 105.0, RATE, 1.0, 5)
except ValueError as e:
    print('rejected as expected:', e)


# ## Conclusions
# * For a fixed number of steps there is a 1-1 correspondence between the up step size and the price: a higher market price (higher implied volatility) needs a larger up step.
# * With more steps the calibrated step size gets smaller, and the equivalent volatility of the tree converges to the volatility behind the market price.
# * For a one-step tree the calibrated step depends on strike and expiry (a "smile" of step sizes), even though all prices come from the same Black-Scholes volatility. This shows that a one-step tree is a very crude model.
# * A price is only calibratable if it lies between the zero-volatility price (`max(S - K*df, 0)`) and the price at the largest admissible step; otherwise `calibrate` raises a clear error.

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections.abc import Callable
from typing import Dict, List, Union
from scipy.stats import norm

def european_call_payoff(S: float, K: float) -> float:
    return max(S-K, 0.0)

def create_spot_tree(spot: float, spot_mult_up: float, spot_mult_down: float, steps: int) -> list[list[float]]:
    previous_level = [spot]
    tree = [previous_level]
    for _ in range(steps):
        new_level = [s * spot_mult_down for s in previous_level]
        new_level += [previous_level[-1] * spot_mult_up]
        tree += [new_level]
        previous_level = new_level
    return tree

def create_discounted_price_tree(spot_tree: list[list[float]], discount_factor: float, K: float, diag: int = 0) -> list[list[float]]:
    spot = spot_tree[0][0]
    spot_mult_up = spot_tree[1][-1]
    spot_mult_down = spot_tree[1][0]
    p_up = ((1 / discount_factor - spot_mult_down) /
                   (spot_mult_up - spot_mult_down))
    p_down = 1 - p_up
    steps = len(spot_tree) - 1
    continuation_value_tree = [[np.nan for _ in level] for level in spot_tree]
    if diag > 0:
        print("risk-neutral measure: ")
        print(('%.3f' % p_up, '%.3f' % p_down))
        # init delta tree
        delta_tree = [[np.nan for _ in level] for level in spot_tree[:-1]] #delta makes no sense for leaves
    # going backwards, payoff is known in leaves
    for i in range(len(spot_tree[-1])):
        spot = spot_tree[-1][i]
        discounted_continuation_value = discount_factor**(steps) * european_call_payoff(spot, K)
        continuation_value_tree[-1][i] = discounted_continuation_value
    for step in range(steps - 1, -1, -1):
        for i in range(len(spot_tree[step])):
            continuation_value_tree[step][i] = p_up * continuation_value_tree[step + 1][i] + \
                                            p_down * continuation_value_tree[step + 1][i + 1]
            if diag > 0:
                delta_tree[step][i] = ((continuation_value_tree[step + 1][i] - continuation_value_tree[step + 1][i + 1]) 
                                       / (spot_tree[step + 1][i] - spot_tree[step + 1][i + 1]))
    if diag > 0:
        print("delta: ")
        delta_tree_readable = [['%.3f' % e for e in n] for n in delta_tree]
        print(delta_tree_readable)
    return continuation_value_tree

def calcBalancedDownStep(spot_mult_up: float, discount_factor: float) -> (float, float):
    return spot_mult_up - 2 * (spot_mult_up - 1 / discount_factor)

def up_and_out(B: float) -> Callable[float, float]:
    def knock_multiplier(s: float) -> float:
        return 1.0 if (s < B) else 0.0
    return knock_multiplier

def create_discounted_price_tree_ko(spot_tree: list[list[float]], discount_factor: float, K: float, ko_function: Callable[float, float], diag: int = 0) -> list[list[float]]:
    spot = spot_tree[0][0]
    spot_mult_up = spot_tree[1][-1]
    spot_mult_down = spot_tree[1][0]
    p_up = ((1 / discount_factor - spot_mult_down) /
                   (spot_mult_up - spot_mult_down))
    p_down = 1 - p_up
    steps = len(spot_tree) - 1
    continuation_value_tree = [[np.nan for _ in level] for level in spot_tree]
    if diag > 0:
        print("risk-neutral measure: ")
        print((p_up, p_down))
        # init delta tree
        delta_tree = [[np.nan for _ in level] for level in spot_tree[:-1]] #delta makes no sense for leaves
    # going backwards, payoff is known in leaves
    for i in range(len(spot_tree[-1])):
        spot = spot_tree[-1][i]
        discounted_continuation_value = discount_factor**(steps) * european_call_payoff(spot, K) * ko_function(spot)
        continuation_value_tree[-1][i] = discounted_continuation_value
    for step in range(steps - 1, -1, -1):
        for i in range(len(spot_tree[step])):
            continuation_value_tree[step][i] = p_up * continuation_value_tree[step + 1][i] + \
                                            p_down * continuation_value_tree[step + 1][i + 1]
            continuation_value_tree[step][i] *= ko_function(spot_tree[step][i])
            if diag > 0:
                delta_tree[step][i] = ((continuation_value_tree[step + 1][i] - continuation_value_tree[step + 1][i + 1]) 
                                       / (spot_tree[step + 1][i] - spot_tree[step + 1][i + 1]))
                delta_tree[step][i] *= ko_function(spot_tree[step][i])
    if diag > 0:
        print("delta: ")
        delta_tree_readable = [['%.3f' % e for e in n] for n in delta_tree]
        print(delta_tree_readable)
    return continuation_value_tree

def black_scholes_eur_call(r: float, T: float, S0: float, sigma: float, K: Union[float, List[float]]):
    assert sigma > 0

    K = np.array([K]) if isinstance(K, float) else np.array(K)

    d1_vec = ( np.log( S0 / K ) + ( r + 0.5 * sigma**2 ) * T ) / ( sigma * T**0.5 )
    d2_vec = d1_vec - sigma * T**0.5

    N_d1_vec = norm.cdf(d1_vec)
    N_d2_vec = norm.cdf(d2_vec)
    c=N_d1_vec * S0 - K * np.exp((-1.0)*r*T) * N_d2_vec
    

    return {'call_price': c}

def price_from_up_step(spot_mult_up: float, steps: int, df: float, spot: float, strike: float):
    discount_factor_per_step = df ** (1/steps)
    spot_mult_down = calcBalancedDownStep(spot_mult_up, discount_factor_per_step)
    spot_tree = create_spot_tree(spot, spot_mult_up, spot_mult_down, steps)
    price_tree = create_discounted_price_tree(spot_tree, discount_factor_per_step, strike)
    return price_tree[0][0]

def calibrate(target_price: float, steps: int, df: float, spot: float, strike: float, sigma: float):
    def objective(spot_mult_up):
        price = price_from_up_step(spot_mult_up[0], steps, df, spot, strike)
        return (price - target_price) ** 2
    
    x0 = [np.exp(sigma * np.sqrt(1/steps))]
    result = minimize(objective, x0=x0, bounds=[(1.001, None)])
    return result.x[0]

spot   = 1.0
sigma  = 0.2
r      = 0.05
strike = 1.0
T      = 1.0

df = np.exp(-r * T)
target_price = black_scholes_eur_call(r, T, spot, sigma, strike)['call_price'][0]
print(f"BS target price: {target_price:.4f}")

steps_list = [1, 2, 5, 10, 20, 50]
calibrated_up_steps = []

for steps in steps_list:
    up = calibrate(target_price, steps, df, spot, strike, sigma=sigma)
    calibrated_up_steps.append(up)
    print(f"steps={steps:3d}  spot_mult_up={up:.6f}")

plt.figure(figsize=(7, 4))
plt.plot(steps_list, calibrated_up_steps, marker='o')
plt.grid(True)
plt.tight_layout()
plt.show()
steps      = 1
strikes    = [0.8, 0.9, 1.0, 1.1, 1.2]
maturities = [0.25, 0.5, 1.0, 2.0]

plt.figure(figsize=(7, 4))

for T in maturities:
    df = np.exp(-r * T)
    up_steps = []
    for strike in strikes:
        target_price = black_scholes_eur_call(r, T, spot, sigma, strike)['call_price'][0]
        up = calibrate(target_price, steps, df, spot, strike, sigma=sigma)
        up_steps.append(up)
    plt.plot(strikes, up_steps, marker='o', label=f"T={T}")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
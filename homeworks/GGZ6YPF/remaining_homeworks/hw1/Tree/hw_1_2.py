import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def european_call_payoff(S: float, K: float) -> float:
    return max(S - K, 0.0)

def calcBalancedDownStep(spot_mult_up: float, discount_factor: float) -> float:
    return spot_mult_up - 2 * (spot_mult_up - 1 / discount_factor)

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

spot = 1
spot_mult_up = 2
spot_mult_down = 0.5
steps = 2
risk_free_rate = 0.03
spot_tree = create_spot_tree(spot, spot_mult_up, spot_mult_down, steps)
spot_tree_readable = [['%.3f' % e for e in n] for n in spot_tree]
print(spot_tree_readable)
discount_factor = 0.95
strike = 1
diag = 1
price_tree = create_discounted_price_tree(spot_tree, discount_factor, strike, diag)
price_tree_readable = [['%.3f' % e for e in n] for n in price_tree]
print("Price tree:")
print(price_tree_readable)

def binomial_tree_option_price(spot: float, strike: float, steps: int, risk_free_rate: float, spot_mult_up: float) -> float:
    discount_factor = 1 / (1 + risk_free_rate)
    spot_mult_down = calcBalancedDownStep(spot_mult_up, discount_factor)
    spot_tree = create_spot_tree(spot, spot_mult_up, spot_mult_down, steps)
    for i in range(len(spot_tree[steps])):
        spot_tree[steps][i] = european_call_payoff(spot_tree[steps][i], strike)
    print(spot_tree)    
    price_tree = create_discounted_price_tree(spot_tree, discount_factor, strike)
    return price_tree[0][0]

binomial_tree_option_price(spot, strike, steps, risk_free_rate, spot_mult_up)

def calibrate_up_step(spot: float, strike: float, maturity: float, market_price: float, steps_per_year: int, risk_free_rate: float) -> float:
    
    def objective(spot_mult_up: float) -> float:
        model_price = binomial_tree_option_price(spot, strike, steps, risk_free_rate, spot_mult_up)
        return (model_price - market_price) ** 2

    result = minimize(objective, x0=1.1, bounds=[(0.01, 5)])
    return result.x[0]
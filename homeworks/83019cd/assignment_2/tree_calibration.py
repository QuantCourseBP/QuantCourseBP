"""Group 1, Assignment 2: balanced-tree calibration.

Based on the functions in the 2026 BME Binomial_tree_coding.ipynb.
A step is a multiplicative factor, not a currency increment.
"""
import numpy as np
from scipy.optimize import brentq


def european_call_payoff(S: float, K: float) -> float:
    return max(S - K, 0.0)


def _finite_scalar(name, value):
    if np.ndim(value) != 0 or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite scalar.')


def _validate_contract(spot, steps, K, expiry, rate):
    for name, value in [('spot', spot), ('K', K), ('expiry', expiry), ('rate', rate)]:
        _finite_scalar(name, value)
    if spot <= 0 or K <= 0 or expiry <= 0:
        raise ValueError('spot, K and expiry must be positive.')
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError('steps must be a positive integer.')


def calcBalancedDownStep(spot_mult_up: float, discount_factor: float) -> float:
    """Lecture formula: d = 2 / DF_step - u, so risk-neutral p_up = 0.5."""
    return 2 / discount_factor - spot_mult_up


def create_spot_tree(spot: float, spot_mult_up: float, spot_mult_down: float,
                     steps: int) -> list[list[float]]:
    """Lecture's recombining tree, ordered from lowest to highest node."""
    previous_level = [spot]
    tree = [previous_level]
    for _ in range(steps):
        new_level = [s * spot_mult_down for s in previous_level]
        new_level += [previous_level[-1] * spot_mult_up]
        tree += [new_level]
        previous_level = new_level
    return tree


def create_discounted_price_tree(spot_tree: list[list[float]], discount_factor: float,
                                 K: float, diag: int = 0) -> list[list[float]]:
    """Lecture convention: every node value is already discounted to time zero.

    Normalize first-level spots to obtain multipliers (the lecture example
    uses spot=1). Children are ordered down, up, so use p_down then p_up.
    The deterministic u=d case is allowed to evaluate the calibration bound.
    """
    _finite_scalar('discount_factor', discount_factor)
    if discount_factor <= 0:
        raise ValueError('discount_factor must be positive.')
    spot = spot_tree[0][0]
    up = spot_tree[1][-1] / spot
    down = spot_tree[1][0] / spot
    growth = 1 / discount_factor
    if up == down:
        if not np.isclose(up, growth, rtol=1e-12, atol=0):
            raise ValueError('A deterministic tree must grow at the risk-free rate.')
        p_up = 0.5
    else:
        p_up = (growth - down) / (up - down)
    if not 0 <= p_up <= 1:
        raise ValueError('Tree factors do not admit risk-neutral probabilities.')
    p_down = 1 - p_up
    steps = len(spot_tree) - 1
    tree = [[np.nan for _ in level] for level in spot_tree]
    tree[-1] = [discount_factor**steps * european_call_payoff(s, K) for s in spot_tree[-1]]
    for step in range(steps - 1, -1, -1):
        for i in range(len(spot_tree[step])):
            tree[step][i] = p_down * tree[step + 1][i] + p_up * tree[step + 1][i + 1]
    if diag > 0:
        print(f'Risk-neutral probabilities: up={p_up:.6f}, down={p_down:.6f}')
    return tree


def balanced_call_price(spot: float, spot_mult_up: float, steps: int, K: float,
                        expiry: float, rate: float) -> float:
    """Price with p_up=p_down=0.5 and a constant continuously compounded rate.

    u=m and u=2m are included for calibration boundary evaluation. Production
    calibration returns an interior solution with 0 < d < m < u < 2m.
    """
    _validate_contract(spot, steps, K, expiry, rate)
    _finite_scalar('spot_mult_up', spot_mult_up)
    discount_factor = np.exp(-rate * expiry / steps)
    growth = 1 / discount_factor
    if not growth <= spot_mult_up <= 2 * growth:
        raise ValueError('Balanced tree requires growth <= up <= 2*growth.')
    down = calcBalancedDownStep(spot_mult_up, discount_factor)
    spot_tree = create_spot_tree(spot, spot_mult_up, down, steps)
    # Balancing fixes probabilities exactly, including near the deterministic
    # limit where extracting them by subtracting near-equal factors is unstable.
    values = np.maximum(np.asarray(spot_tree[-1]) - K, 0.)
    for _ in range(steps):
        values = discount_factor * 0.5 * (values[:-1] + values[1:])
    return float(values[0])


def calibrate(observed_price: float, steps: int, K: float, expiry: float,
              rate: float, spot: float) -> float:
    """Return the balanced up multiplier reproducing a European call price.

    Solve price(u)-observed_price=0 using scipy.optimize.brentq, a bracketed
    root solver. Parameterize u=m*(1+a), d=m*(1-a), 0<=a<=1, m=exp(r*T/n).
    We require a positive down factor and a uniquely identifiable step:
    prices equal to the deterministic minimum or zero-down upper boundary
    are rejected, as are prices outside this finite tree's attainable range.
    """
    _validate_contract(spot, steps, K, expiry, rate)
    _finite_scalar('observed_price', observed_price)
    growth = 1 / np.exp(-rate * expiry / steps)
    minimum = balanced_call_price(spot, growth, steps, K, expiry, rate)
    maximum = balanced_call_price(spot, 2 * growth, steps, K, expiry, rate)
    tolerance = 1e-12 * max(1., spot, abs(observed_price))
    if observed_price < minimum - tolerance or observed_price > maximum + tolerance:
        raise ValueError(f'Price outside attainable range [{minimum:.12g}, {maximum:.12g}].')
    if observed_price <= minimum + tolerance:
        raise ValueError('Price is at the deterministic minimum; the step is not uniquely identifiable.')
    if observed_price >= maximum - tolerance:
        raise ValueError('Price is at the upper boundary; no interior positive-down calibration exists.')

    def residual(amplitude):
        up = growth * (1 + amplitude)
        return balanced_call_price(spot, up, steps, K, expiry, rate) - observed_price

    amplitude = brentq(residual, 0., 1., xtol=1e-14, rtol=1e-14)
    up = growth * (1 + amplitude)
    if abs(residual(amplitude)) > tolerance:
        raise RuntimeError('Root solver did not reproduce the price within tolerance.')
    return float(up)

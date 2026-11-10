"""Analytic, two-dimensional finite difference and exact-terminal MC pricing.

Model: independent risk-neutral GBMs, equal constant volatility, no dividends.
All prices are per contract, in the same currency as the two stock prices.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.stats import norm


def _validate(spot1, spot2, volatility, maturity, rate=0.):
    if not all(np.isfinite(x) for x in (spot1, spot2, volatility, maturity, rate)):
        raise ValueError('Model inputs must be finite')
    if spot1 <= 0 or spot2 <= 0:
        raise ValueError('Stock prices must be positive')
    if volatility < 0 or maturity < 0:
        raise ValueError('Volatility and remaining maturity must be nonnegative')


def _positive_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def analytic_price(spot1=100., spot2=100., volatility=.2, maturity=1., rate=.05):
    """Exchange-option identity; rate cancels under the no-dividend assumptions."""
    _validate(spot1, spot2, volatility, maturity, rate)
    if volatility == 0 or maturity == 0:
        return float(max(spot1, spot2))
    total_vol = np.sqrt(2.) * volatility * np.sqrt(maturity)
    log_ratio = np.log(spot1) - np.log(spot2)
    d1 = log_ratio / total_vol + .5 * total_vol
    d2 = -log_ratio / total_vol + .5 * total_vol
    return float(spot1 * norm.cdf(d1) + spot2 * norm.cdf(d2))


@dataclass(frozen=True)
class PDEResult:
    price: float
    spot1_grid: np.ndarray
    spot2_grid: np.ndarray
    values: np.ndarray
    space_steps: int
    time_steps: int
    log_half_width: float
    runtime_seconds: float


def finite_difference_price(spot1=100., spot2=100., volatility=.2, maturity=1., rate=.05,
                            space_steps=100, time_steps=200, log_half_width=1.5):
    """Solve the full two-stock PDE on a rectangular log-price grid.

    x=log(S1/spot1), y=log(S2/spot2), tau=T-t. Central spatial differences,
    Crank-Nicolson time stepping and two initial implicit half steps
    (Rannacher smoothing). Even space_steps place requested spots on a node.

    At the artificial outer boundary use V ~= max(S1,S2), the dominant-stock
    asymptote. It is NOT exact near equal-stock corners. Increase domain width
    at fixed spacing to check truncation error, separately from grid error.
    No analytic prices are used in the solver or its boundary values.
    """
    start = perf_counter()
    _validate(spot1, spot2, volatility, maturity, rate)
    _positive_integer(space_steps, 'space_steps', 4)
    _positive_integer(time_steps, 'time_steps', 1)
    if space_steps % 2:
        raise ValueError('space_steps must be even so the requested spots lie on a node')
    if not np.isfinite(log_half_width) or not 0 < log_half_width <= 20:
        raise ValueError('log_half_width must be finite and in (0, 20]')
    coordinates = np.linspace(-log_half_width, log_half_width, space_steps + 1)
    s1, s2 = spot1 * np.exp(coordinates), spot2 * np.exp(coordinates)
    values = np.maximum(s1[:, None], s2[None, :])
    if volatility > 0 and maturity > 0:
        dx = 2 * log_half_width / space_steps
        dt = maturity / time_steps
        diffusion = .5 * volatility**2
        drift = rate - .5 * volatility**2
        lower = diffusion / dx**2 - drift / (2*dx)
        upper = diffusion / dx**2 + drift / (2*dx)
        if min(lower, upper) < 0:
            raise ValueError('Central drift stencil is non-monotone; refine the spatial grid')
        size = space_steps - 1
        one_d = diags([np.full(size-1, lower), np.full(size, -2*diffusion/dx**2),
                       np.full(size-1, upper)], [-1, 0, 1], format='csc')
        identity = eye(size, format='csc')
        operator = (kron(one_d, identity, format='csc') +
                    kron(identity, one_d, format='csc') - rate*eye(size*size, format='csc'))
        # Fixed Dirichlet boundary contributions in the semidiscrete ODE u'=Au+b.
        boundary = np.zeros((size, size))
        boundary[0, :] += lower * values[0, 1:-1]
        boundary[-1, :] += upper * values[-1, 1:-1]
        boundary[:, 0] += lower * values[1:-1, 0]
        boundary[:, -1] += upper * values[1:-1, -1]
        b = boundary.ravel()
        state = values[1:-1, 1:-1].ravel().copy()
        unit = eye(size*size, format='csc')
        # The same factorization serves both half implicit steps and CN steps.
        solve = splu(unit - .5*dt*operator).solve
        for _ in range(2):
            state = solve(state + .5*dt*b)
        rhs_operator = unit + .5*dt*operator
        for _ in range(time_steps-1):
            state = solve(rhs_operator @ state + dt*b)
        values[1:-1, 1:-1] = state.reshape(size, size)
    center = space_steps // 2
    return PDEResult(float(values[center, center]), s1, s2, values,
                     space_steps, time_steps, log_half_width, perf_counter()-start)


@dataclass(frozen=True)
class MCResult:
    price: float
    standard_error: float
    ci_low: float
    ci_high: float
    num_paths: int
    seed: int
    runtime_seconds: float


def monte_carlo_price(spot1=100., spot2=100., volatility=.2, maturity=1., rate=.05,
                      num_paths=100_000, seed=2026):
    """Independent exact GBM terminal draws; ordinary sample-mean 95% CLT CI.

    No time discretization, analytic price, control variates or antithetic
    dependence are used. The SE uses the unbiased sample variance (ddof=1).
    """
    start = perf_counter()
    _validate(spot1, spot2, volatility, maturity, rate)
    _positive_integer(num_paths, 'num_paths', 2)
    _positive_integer(seed, 'seed', 0)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((num_paths, 2))
    # Combine discounting with the exact exponential to reduce intermediate scale.
    discounted_stocks = np.array([spot1, spot2]) * np.exp(
        -.5*volatility**2*maturity + volatility*np.sqrt(maturity)*z)
    payoffs = discounted_stocks.max(axis=1)
    price = float(payoffs.mean())
    se = float(payoffs.std(ddof=1) / np.sqrt(num_paths))
    return MCResult(price, se, price - 1.96*se, price + 1.96*se,
                    num_paths, seed, perf_counter()-start)

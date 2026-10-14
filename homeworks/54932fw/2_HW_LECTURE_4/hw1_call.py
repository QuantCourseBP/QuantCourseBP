import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Dict, List, Union

"""The pricing function of European call option"""
# change # 2
#Delete.    def black_scholes_eur_call(r: float, T: float, S0: float, sigma: float,
#Delete.                               K: Union[float, List[float], np.ndarray]) -> np.ndarray:
# replace by
def black_scholes_eur_call(r: float, T: float, S0: float, sigma: float,
                           K: Union[float, List[float], np.ndarray],
                           return_greeks: bool = False):
# end change # 2
    """
    Black-Scholes pricer of European call option on non-dividend-paying stock

    param r: risk-free interest rate (which is constant)
    param T: time to maturity (in years)
    param S0: initial spot price of the underlying stock
    param sigma: volatility of the underlying stock
    param K: strike price (or prices)
    param return_greeks: if True, return a dict with the price and the greeks
    """
    # check conditions
    assert sigma > 0

    K = np.array([K]) if isinstance(K, float) else np.array(K)

    d1_vec = ( np.log( S0 / K ) + ( r + 0.5 * sigma**2 ) * T ) / ( sigma * T**0.5 )
    d2_vec = d1_vec - sigma * T**0.5

    N_d1_vec = norm.cdf(d1_vec)
    N_d2_vec = norm.cdf(d2_vec)

    # change # 1
    #Delete.    return N_d1_vec * S0 - K_vec * np.exp((-1.0)*r*T) * N_d2_vec
    # replace by
    price = N_d1_vec * S0 - K * np.exp((-1.0)*r*T) * N_d2_vec
    # end change # 1

    # change # 2
    if not return_greeks:
        return price
    n_d1_vec = norm.pdf(d1_vec)                  # N'(d1)
    return {
        "price": price,
        "delta": N_d1_vec,
        "gamma": n_d1_vec / (S0 * sigma * T**0.5),
        "vega":  S0 * n_d1_vec * T**0.5,
        "theta": -S0 * n_d1_vec * sigma / (2 * T**0.5) - r * K * np.exp(-r*T) * N_d2_vec,
        "rho":   K * T * np.exp(-r*T) * N_d2_vec,
    }
    # end change # 2

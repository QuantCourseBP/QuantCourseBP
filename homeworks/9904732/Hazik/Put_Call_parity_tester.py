import numpy as np
from scipy.stats import norm


def black_scholes_eur_call(r, T, S0, sigma, K):
    """
    Black-Scholes price for a European call option

    r: risk-free interest rate
    T: time to maturity (years)
    S0: current stock price
    sigma: volatility
    K: strike price (can be a single value or list)
    """

    #Convert K to numpy array so we can handle multiple strikes
    if isinstance(K, (int, float)):
        K = np.array([K])
    else:
        K = np.array(K)

    #Black-Scholes d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    #Call price
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return price

def black_scholes_eur_put(r, T, S0, sigma, K):
    """
    Black-Scholes price for a European put option

    r: risk-free interest rate
    T: time to maturity (years)
    S0: current stock price
    sigma: volatility
    K: strike price (can be a single value or list)
    """

    #Convert K to numpy array so we can handle multiple strikes
    if isinstance(K, (int, float)):
        K = np.array([K])
    else:
        K = np.array(K)

    #Black-Scholes d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    #Put price
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return price

r = 0.05      #5% risk-free rate
T = 1.0       #1 year to maturity
S0 = 100      #stock price
sigma = 0.2   #20% volatility
K = 100       #strike price

price_call = black_scholes_eur_call(r, T, S0, sigma, K)
price_put = black_scholes_eur_put(r, T, S0, sigma, K)

if (price_call-price_put-(S0-K * np.exp(-r * T)) < np.exp(-10)):
    print("Put-Call parity holds.")
else:
    print("Put-Call parity doesn't hold.")
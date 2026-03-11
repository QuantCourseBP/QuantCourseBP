import numpy as np
from scipy.stats import norm


def black_scholes_eur_put(r, T, S0, sigma, K):
    """
    Black-Scholes price and Greeks for a European put option

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

    #Greeks
    delta = norm.cdf(d1) - 1
    vega = S0 * norm.pdf(d1) * np.sqrt(T)
    theta = -(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return price, delta, vega, theta, gamma, rho

r = 0.05      #5% risk-free rate
T = 1.0       #1 year to maturity
S0 = 100      #stock price
sigma = 0.2   #20% volatility
K = 100       #strike price

price, delta, vega, theta, gamma, rho = black_scholes_eur_put(r, T, S0, sigma, K)

print("European Put Option Results")
print("Price:", price)
print("Delta:", delta)
print("Vega:", vega)
print("Theta:", theta)
print("Gamma:", gamma)
print("Rho:", rho)
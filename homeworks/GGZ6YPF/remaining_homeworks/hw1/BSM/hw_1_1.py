import matplotlib.pyplot as plt
from matplotlib import rcParams # detailed parameter setting
from typing import Dict, List, Union
import numpy as np
from scipy.stats import norm

# a)
def my_plotter(x: List[float], y: Union[List[float], List[List[float]]], layout: Dict = {}, names: List[str] = None):
    """ inline for loop is called 'list comprehension' """
    y = [y] if all(isinstance(item, float) for item in y) else y
    
    plt.figure(figsize=(8, 4))
    lines = []
    show_legend = True if names is not None else False
    # show_legend = True if names else False  -> here is fine, but be cautious bc 0, empty str and empty list will be evaulated to False

    """ setup some basic key-word arguments for plot line """
    plot_kwargs = {
        'linestyle': 'solid',
        'linewidth': 4
    }
    if names is not None:
        show_legend = True
        if len(names) != len(y):
            raise ValueError("Length of names is not matching with number of plotted y lists.")

    """ 'enumerate' add a counter to the loop """
    for i, y_item in enumerate(y):
        if show_legend:
            plot_kwargs['label'] = names[i]
        _line = plt.plot(x, y_item, **plot_kwargs)
        lines.append(_line)

    if show_legend:
        plt.legend(fontsize=16)
    if 'title' in layout:
        plt.title(layout['title'], fontsize=20)
        rcParams['axes.titlepad'] = 20
    if 'x_label' in layout:
        plt.xlabel(layout['x_label'], fontsize=16)
        rcParams['axes.labelpad'] = 20
    if 'y_label' in layout:
        plt.ylabel(layout['y_label'], fontsize=16)
        rcParams['axes.labelpad'] = layout.get('y_label_pad', 20)
    if 'xlim' in layout:
        plt.xlim(layout['xlim'])
    if 'ylim' in layout:
        plt.ylim(layout['ylim'])
    if 'grid' in layout:
        plt.grid(layout['grid'], linestyle='--', linewidth=1, alpha=0.7)

    """ enhance axes """
    ax = plt.gca() # gca: get current axes
    ax.axhline(linestyle='--', color='black', linewidth=1)
    plt.show()

# Example usage
x = np.linspace(-10, 10, 200)
y1 = np.sin(x)
y2 = np.cos(x)

layout = {
    'title': 'Sine and Cosine Waves',
    'x_label': 'Variable',
    'y_label': 'Functions',
    'xlim': [-11, 11],
    'ylim': [-1.5, 1.5],
    'grid': True,
    'title_pad': 10,
    'x_label_pad': 15,
    'y_label_pad': 15
}

names = ['Sine', 'Cosine']

my_plotter(x, [y1, y2], layout, names)

# b)
"""The function of European call option price and greeks"""
def black_scholes_eur_call_greeks(r: float, T: float, S0: float, sigma: float, K: Union[float, List[float]]):
    """
    param r: risk-free interest rate (which is constant)
    param T: time to maturity (in years)
    param S0: initial spot price of the underlying stock
    param sigma: volatility of the underlying stock
    param K: strike price (or prices)
    """
    # check conditions
    assert sigma > 0

    K = np.array([K]) if isinstance(K, float) else np.array(K)

    d1_vec = ( np.log( S0 / K ) + ( r + 0.5 * sigma**2 ) * T ) / ( sigma * T**0.5 )
    d2_vec = d1_vec - sigma * T**0.5

    N_d1_vec = norm.cdf(d1_vec)
    N_d2_vec = norm.cdf(d2_vec)
    N_pdf_d1_vec = norm.pdf(d1_vec)

    price = N_d1_vec * S0 - K * np.exp((-1.0)*r*T) * N_d2_vec
    delta = N_d1_vec
    vega = S0 * N_pdf_d1_vec * np.sqrt(T)
    theta = -(S0 * N_pdf_d1_vec * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2_vec
    rho = K * T * np.exp(-r * T) * N_d2_vec
    gamma = N_pdf_d1_vec / (S0 * sigma * np.sqrt(T))

    return price, delta, vega, theta, rho, gamma

price, delta, vega, theta, rho, gamma = black_scholes_eur_call_greeks(r = 0.05, T = 2.0, S0 = 20.0 , sigma = 0.3, K = 30.0)
print(f"Price: {price[0]}, delta: {delta[0]}, vega: {vega[0]}, theta: {theta[0]}, rho: {rho[0]}, gamma: {gamma[0]}")

# c)
"""The pricing function of European call option"""
def black_scholes_eur_call(r: float, T: float, S0: float, sigma: float, K: Union[float, List[float]]):
    """
    Black-Scholes pricer of European call option on non-dividend-paying stock

    param r: risk-free interest rate (which is constant)
    param T: time to maturity (in years)
    param S0: initial spot price of the underlying stock
    param sigma: volatility of the underlying stock
    param K: strike price (or prices)
    """
    # check conditions
    assert sigma > 0

    K = np.array([K]) if isinstance(K, float) else np.array(K)

    d1_vec = ( np.log( S0 / K ) + ( r + 0.5 * sigma**2 ) * T ) / ( sigma * T**0.5 )
    d2_vec = d1_vec - sigma * T**0.5

    N_d1_vec = norm.cdf(d1_vec)
    N_d2_vec = norm.cdf(d2_vec)

    return N_d1_vec * S0 - K * np.exp((-1.0)*r*T) * N_d2_vec

"""The pricing function of European put option"""
def black_scholes_eur_put(r: float, T: float, S0: float, sigma: float, K: Union[float, List[float]]):
    """
    Black-Scholes pricer of European put option on non-dividend-paying stock

    param r: risk-free interest rate (which is constant)
    param T: time to maturity (in years)
    param S0: initial spot price of the underlying stock
    param sigma: volatility of the underlying stock
    param K: strike price (or prices)
    """
    # check conditions
    assert sigma > 0

    K = np.array([K]) if isinstance(K, float) else np.array(K)

    d1_vec = ( np.log( S0 / K ) + ( r + 0.5 * sigma**2 ) * T ) / ( sigma * T**0.5 )
    d2_vec = d1_vec - sigma * T**0.5

    N_neg_d1_vec = norm.cdf(-d1_vec)
    N_neg_d2_vec = norm.cdf(-d2_vec)

    return K * np.exp((-1.0)*r*T) * N_neg_d2_vec - N_neg_d1_vec * S0

# strike prices for which the option price is calculated
K_vec = np.arange(10, 30, 0.01)
# time to maturities (in year fractions)
T_vec = [1.0, 2.0, 5.0]

prices_to_plot = []
for _T in T_vec:
    prices_to_plot.append(
        black_scholes_eur_call(r = 0.05, T = _T, S0 = 20.0 , sigma = 0.3, K = K_vec)
    )

layout = {
    'title': 'Price of a European Call ($S_0 = 20$, $\sigma = 0.3$, $r = 0.05$)\n$T$: time to maturity in years'
}
my_plotter(K_vec, prices_to_plot, layout=layout, names=[f'T = {int(item)}' for item in T_vec])

# strike prices for which the option price is calculated
K_vec = np.arange(10, 30, 0.01)
# time to maturities (in year fractions)
T_vec = [1.0, 2.0, 5.0]

prices_to_plot = []
for _T in T_vec:
    prices_to_plot.append(
        black_scholes_eur_put(r = 0.05, T = _T, S0 = 20.0 , sigma = 0.3, K = K_vec)
    )

layout = {
    'title': 'Price of a European Put ($S_0 = 20$, $\sigma = 0.3$, $r = 0.05$)\n$T$: time to maturity in years'
}
my_plotter(K_vec, prices_to_plot, layout=layout, names=[f'T = {int(item)}' for item in T_vec])

# d)
"""The pricing function of a forward"""
def forward(r: float, T: float, S0: float, K: float):
    
    """
    param r: risk-free interest rate (which is constant)
    param T: time to maturity (in years)
    param S0: initial spot price of the underlying stock
    param K: strike price
    """

    return S0 - K * np.exp((-1.0)*r*T)

# Put-Call parity checker
r_vec = [0.02, 0.05, 0.08]
T_vec = [1.0, 2.0, 5.0]
S0_vec = [10.0, 20.0, 30.0]
sigma_vec = [0.1, 0.3, 0.5]
K_vec = [10.0, 20.0, 30.0]

def put_call_parity_checker(r_vec, T_vec, S0_vec, sigma_vec, K_vec, threshold = 1e-14):
    inst = 0 # counts the number of cases when Put-Call parity doesn't hold
    for _r in r_vec:
        for _T in T_vec:
            for _S0 in S0_vec:
                for _sigma in sigma_vec:
                    for _K in K_vec:
                        diff = black_scholes_eur_call(r = _r, T = _T, S0 = _S0, sigma = _sigma, K = _K) - black_scholes_eur_put(r = _r, T = _T, S0 = _S0, sigma = _sigma, K = _K) - forward(r = _r, T = _T, S0 = _S0, K = _K)
                        if abs(diff) > threshold:
                            inst += 1
                            print(f"Put-Call parity does not hold with parameters r = {_r}, T = {_T}, S0 = {_S0}, sigma = {_sigma}, K = {_K}.")
    if inst == 0:
        print("Put-Call parity holds.")

p_c_check = put_call_parity_checker(r_vec, T_vec, S0_vec, sigma_vec, K_vec)
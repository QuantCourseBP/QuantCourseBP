import matplotlib.pyplot as plt
from matplotlib import rcParams # detailed parameter setting
from typing import Dict, List, Union
import numpy as np
from scipy.stats import norm

"""A Smarted up plotter """
def my_plotter(x: List[float], y: Union[List[float], List[List[float]]], layout: Dict = {}, names: List[str] = None, colours: List[str]=None, linestyle: str = 'solid', linewidth: int = 4, figsize: tuple = (8,4), legend_fontsize: int = 16, title_fontsize: int = 20, x_fontsize: int = 16, titlepad: int = 30, labelpad: int = 20):
    """ inline for loop is called 'list comprehension' """
    y = [y] if all(isinstance(item, float) for item in y) else y
    
    plt.figure(figsize=figsize)
    lines = []
    show_legend = True if names is not None else False
    # show_legend = True if names else False  -> here is fine, but be cautious bc 0, empty str and empty list will be evaulated to False

    """ setup some basic key-word arguments for plot line """
    plot_kwargs = {
        'linestyle': linestyle,
        'linewidth': linewidth
    }
    if names is not None:
        show_legend = True
        if len(names) != len(y):
            raise ValueError("Length of names is not matching with number of plotted y lists.")
    if len(colours)!=len(y):
        raise ValueError("Length of names is not matching with number of plotted y lists.")
        

    """ 'enumerate' add a counter to the loop """
    for i, y_item in enumerate(y):
        if show_legend:
            plot_kwargs['label'] = names[i]
            plot_kwargs['color'] = colours[i]
        _line = plt.plot(x, y_item, **plot_kwargs)
        lines.append(_line)

    if show_legend:
        plt.legend(fontsize=legend_fontsize)
    if 'title' in layout:
        plt.title(layout['title'], fontsize=title_fontsize)
        rcParams['axes.titlepad'] = titlepad # moving the title a little further away from the plot
    if 'x_label' in layout:
        plt.xlabel(layout['x_label'], fontsize=x_fontsize)
        rcParams['axes.labelpad'] = labelpad # moving the ax label a little further away from the plot

    """ enhance axes """
    ax = plt.gca() # gca: get current axes
    ax.axhline(linestyle='--', color='black', linewidth=1)
    plt.show()

"""B The pricing function of European call option, with greeks"""
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
    c=N_d1_vec * S0 - K * np.exp((-1.0)*r*T) * N_d2_vec
    delta=N_d1_vec
    vega= K*np.exp((-1.0)*r*T)*norm.pdf(d2_vec)*np.sqrt(T)
    theta=-1.0*((S0*norm.pdf(d1_vec)*sigma)/(2*np.sqrt(T))+r*K*np.exp((-1.0)*r*T)*norm.cdf(d2_vec))
    gamma=norm.pdf(d1_vec) / (S0 * sigma * np.sqrt(T))
    rho=K*T*np.exp((-1.0)*r*T)*norm.cdf(d2_vec)
    

    return {'call_price': c, 
            'delta': delta,
            'vega': vega, 
            'theta': theta, 
            'gamma': gamma, 
            'rho': rho}

"""C The pricing function of European put option with greeks"""
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

    p=-norm.cdf(-d1_vec) * S0 + K * np.exp((-1.0)*r*T) * norm.cdf(-d2_vec)
    delta=norm.cdf(-d1_vec)
    vega= K*np.exp((-1.0)*r*T)*norm.pdf(d2_vec)*np.sqrt(T)
    theta=-1.0 * ((S0 * norm.pdf(d1_vec) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp((-1.0) * r * T) * norm.cdf(-d2_vec))
    gamma=norm.pdf(d1_vec) / (S0 * sigma * np.sqrt(T))
    rho=-K*T*np.exp((-1.0)*r*T)*norm.cdf(-d2_vec)
    

    return {'put_price': p, 
            'delta': delta,
            'vega': vega, 
            'theta': theta, 
            'gamma': gamma, 
            'rho': rho}


'''D Test the pricer and put-call parity'''
K_vec = np.arange(10, 30, 0.01)
# time to maturities (in year fractions)
T_vec = [1.0, 2.0, 5.0]
K0 = K_vec[0]
T0 = T_vec[0]
r = 0.05
S0 = 20.0
sigma = 0.3

for _T in T_vec:
    call_price = black_scholes_eur_call(r, _T, S0, sigma, K_vec)['call_price']
    put_price = black_scholes_eur_put(r, _T, S0, sigma, K_vec)['put_price']
    
    diff = call_price - put_price
    risk_free = S0 - K_vec * np.exp(-r * _T)
    
    pc_parity = diff - risk_free

    if np.allclose(pc_parity, 0):
        print(f'T={_T}: No arbitrage :(')
    else:
        print(f'T={_T}: Arbitrage?!')
        print(pc_parity)


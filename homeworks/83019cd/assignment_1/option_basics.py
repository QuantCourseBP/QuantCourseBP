"""Assignment 1: extensions of the 2026 BME BSM notebook functions."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def my_plotter(x, y, layout=None, names=None, *, line_kwargs=None, show=True):
    """Plot one series or a list of series; return (figure, axes).

    layout supports figsize, title, x_label, y_label, title_fontsize,
    label_fontsize, titlepad, labelpad, legend_fontsize, legend_loc,
    xlim, ylim, grid and zero_line. line_kwargs accepts Matplotlib line options.
    Settings apply only to this figure, without changing global rcParams.
    """
    layout = {} if layout is None else dict(layout)
    values = np.asarray(y)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != len(x):
        raise ValueError('Each y series must have the same length as x.')
    if names is not None and len(names) != len(values):
        raise ValueError('Length of names is not matching with number of plotted y lists.')
    fig, ax = plt.subplots(figsize=layout.get('figsize', (8, 4)))
    style = {'linestyle': 'solid', 'linewidth': 4}
    if line_kwargs is not None:
        style.update(line_kwargs)
    for i, series in enumerate(values):
        options = dict(style)
        if names is not None:
            options['label'] = names[i]
        ax.plot(x, series, **options)
    ax.set_title(layout.get('title', ''), fontsize=layout.get('title_fontsize', 20),
                 pad=layout.get('titlepad', 30))
    ax.set_xlabel(layout.get('x_label', ''), fontsize=layout.get('label_fontsize', 16),
                  labelpad=layout.get('labelpad', 20))
    ax.set_ylabel(layout.get('y_label', ''), fontsize=layout.get('label_fontsize', 16),
                  labelpad=layout.get('labelpad', 20))
    for key in ('xlim', 'ylim'):
        if key in layout:
            getattr(ax, 'set_' + key)(layout[key])
    ax.grid(layout.get('grid', False))
    if layout.get('zero_line', True):
        ax.axhline(0, linestyle='--', color='black', linewidth=1)
    if names is not None:
        ax.legend(fontsize=layout.get('legend_fontsize', 16), loc=layout.get('legend_loc', 'best'))
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _bsm_inputs(r, T, S0, sigma, K):
    # The lecture assumes a non-dividend-paying stock and positive time/volatility.
    for name, value in [('r', r), ('T', T), ('S0', S0), ('sigma', sigma)]:
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError(f'{name} must be a finite scalar.')
    if T <= 0 or S0 <= 0 or sigma <= 0:
        raise ValueError('T, S0 and sigma must be positive.')
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if K.ndim != 1 or K.size == 0 or np.any(~np.isfinite(K)) or np.any(K <= 0):
        raise ValueError('K must be a positive finite scalar or nonempty one-dimensional array.')
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return K, d1, d1 - sigma * np.sqrt(T)


def black_scholes_eur_call(r, T, S0, sigma, K, return_greeks=False):
    """Return call prices as an array, or a dict of price and five Greeks.

    T is remaining years; r and sigma are decimals. Vega and rho are per
    unit change (divide by 100 for one percentage point). Theta is per year
    of calendar time passing, i.e. minus the derivative with respect to T.
    """
    K, d1, d2 = _bsm_inputs(r, T, S0, sigma, K)
    df = np.exp(-r * T)
    price = S0 * norm.cdf(d1) - K * df * norm.cdf(d2)
    if not return_greeks:
        return price
    return {
        'price': price,
        'delta': norm.cdf(d1),
        'gamma': norm.pdf(d1) / (S0 * sigma * np.sqrt(T)),
        'vega': S0 * norm.pdf(d1) * np.sqrt(T),
        'theta': -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * df * norm.cdf(d2),
        'rho': K * T * df * norm.cdf(d2),
    }


def black_scholes_eur_put(r, T, S0, sigma, K):
    """Direct BSM European put formula, with the call pricer's input convention."""
    K, d1, d2 = _bsm_inputs(r, T, S0, sigma, K)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

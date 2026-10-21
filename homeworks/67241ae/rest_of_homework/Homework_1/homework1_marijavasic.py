import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#(a) A customizable plotter

#The lecture hard-codes figure size, font sizes, line style and width. These are now configurable.
#Additional options include y-axis labels, axis limits, grid, colors, markers, legend position and saving a figure.

def my_plotter(x, y, layout=None, names=None):
    """Plot one curve or a sequence of curves against a shared x array."""
    layout = {} if layout is None else dict(layout)
    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    if x.ndim != 1 or y.ndim != 2 or y.shape[1] != x.size:
        raise ValueError("Use a 1D x array and curves with the same length as x.")
    if names is not None and len(names) != len(y):
        raise ValueError("Provide one name per curve.")
    colors = layout.get("colors")
    if colors is not None and len(colors) < len(y):
        raise ValueError("Provide at least one color per curve.")
    fig, ax = plt.subplots(figsize=layout.get("figsize", (9, 5)))
    for i, curve in enumerate(y):
        ax.plot(x, curve,
                label=None if names is None else names[i],
                color=None if colors is None else colors[i],
                linewidth=layout.get("linewidth", 2),
                linestyle=layout.get("styles", ["-"])[
                    i % len(layout.get("styles", ["-"]))],
                marker=layout.get("marker", None),
                alpha=layout.get("alpha", 1.0))
    ax.set_title(layout.get("title", ""),
                 fontsize=layout.get("title_fontsize", 16),
                 pad=layout.get("titlepad", 15))
    ax.set_xlabel(layout.get("x_label", ""), fontsize=layout.get("label_fontsize", 12))
    ax.set_ylabel(layout.get("y_label", ""), fontsize=layout.get("label_fontsize", 12))
    ax.tick_params(labelsize=layout.get("tick_fontsize", 10))
    ax.grid(layout.get("grid", True), alpha=0.3)
    if layout.get("zero_line", True):
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    if "xlim" in layout:
        ax.set_xlim(layout["xlim"])
    if "ylim" in layout:
        ax.set_ylim(layout["ylim"])
    if names is not None and layout.get("legend", True):
        ax.legend(loc=layout.get("legend_loc", "best"),
                  fontsize=layout.get("legend_fontsize", 10))
    fig.tight_layout()
    if layout.get("save_path"):
        fig.savefig(layout["save_path"], dpi=layout.get("dpi", 150), bbox_inches="tight")
    plt.show()
    return fig, ax

#Test
def test_my_plotter():
    x = np.linspace(-5, 5, 100)

    my_plotter(
        x,
        [x**2, 2*x + 3, np.sin(x)],
        names=["x²", "2x + 3", "sin(x)"],
        layout={
            "title": "Testing my plotter",
            "x_label": "x",
            "y_label": "Function value",
            "figsize": (10, 6),
            "colors": ["#D76A99", "#178B8B", "#7851A9"],
            "linewidth": 2,
            "grid": True,
            "styles": ["-", "--", ":"],
        },
    )

test_my_plotter()



#(b) Call pricing and Greeks

def _bsm_inputs(r, T, S0, sigma, K):
    r, T, S0, sigma, K = np.broadcast_arrays( *[np.asarray(v, dtype=float) for v in (r, T, S0, sigma, K)])
    if not all(np.all(np.isfinite(v)) for v in (r, T, S0, sigma, K)):
        raise ValueError("All inputs must be finite.")
    if any(np.any(v <= 0) for v in (T, S0, sigma, K)):
        raise ValueError("T, S0, sigma and K must be strictly positive.")
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return r, T, S0, sigma, K, d1, d2

def black_scholes_eur_call(r, T, S0, sigma, K, return_greeks=False):
    r, T, S0, sigma, K, d1, d2 = _bsm_inputs(r, T, S0, sigma, K)
    discount = np.exp(-r * T)
    price = S0 * norm.cdf(d1) - K * discount * norm.cdf(d2)
    if not return_greeks:
        return price
    return {
        "price": price,
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S0 * sigma * np.sqrt(T)),
        "vega": S0 * norm.pdf(d1) * np.sqrt(T),
        "theta": -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * discount * norm.cdf(d2),
        "rho": K * T * discount * norm.cdf(d2)
    }
parameters = dict(r=0.05, T=1.0, S0=100.0, sigma=0.20, K=100.0)
call_result = black_scholes_eur_call(**parameters, return_greeks=True)
for name, value in call_result.items():
    print(f"Call {name:>5}: {float(value):.6f}")


#(c) European put pricer

def black_scholes_eur_put(r, T, S0, sigma, K, return_greeks=False):
    r, T, S0, sigma, K, d1, d2 = _bsm_inputs(r, T, S0, sigma, K)
    discount = np.exp(-r * T)
    price = K * discount * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    if not return_greeks:
        return price
    return {
        "price": price,
        "delta": norm.cdf(d1) - 1,
        "gamma": norm.pdf(d1) / (S0 * sigma * np.sqrt(T)),
        "vega": S0 * norm.pdf(d1) * np.sqrt(T),
        "theta": -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))+ r * K * discount * norm.cdf(-d2),
        "rho": -K * T * discount * norm.cdf(-d2)
    }

put_result = black_scholes_eur_put(**parameters, return_greeks=True)
for name, value in put_result.items():
    print(f"{float(value):.6f}")

put_price = black_scholes_eur_put(r=0.05,T=1.0,S0=20.0,sigma=0.3,K=100)
print(put_price)

#(d) Numerical check of put–call parity

r = 0.05
T = 1.0
S0 = 100.0
sigma = 0.20
K = 100.0

call_price = black_scholes_eur_call(r, T, S0, sigma, K)
put_price = black_scholes_eur_put(r, T, S0, sigma, K)

left_side = call_price - put_price
right_side = S0 - K * np.exp(-r * T)

print("Call price:", call_price)
print("Put price:", put_price)
print("C - P:", left_side)
print("S0 - K exp(-rT):", right_side)
print("Difference:", left_side - right_side)

if np.isclose(left_side, right_side):
    print("Put–call parity holds.")
else:
    print("Put–call parity does not hold.")
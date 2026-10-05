"""HW 2/1 -- the examples of the PDF, in the order of the PDF.

Each block below (between two "# %%" lines) is printed in the PDF together
with what it shows.  Paste the blocks into notebook cells, one after the
other, after the code of cell 9 (file professor_cell09.py); or open this file
in VS Code and use "Run Cell".  Run it from this folder (it imports plotter.py
and saves into figures/).
"""
from professor_cell09 import *          # cell 9: the professor's my_plotter

# %% [block 1] cells 10-11 of the notebook, drawn with the plotter of cell 9
import numpy as np

strike = 100
step_size = 0.001

x_stock_price = np.arange(0, 2 * strike, step_size)
y_forward_payoff = np.array([x - strike for x in x_stock_price])

y_forward_payoff__short = y_forward_payoff * (-1)
y_forward_payoff__short

layout = {'title': 'Forward', 'x_label': '$S_{T}$'}
my_plotter(x_stock_price, [y_forward_payoff, y_forward_payoff__short],
           layout=layout,
           names=['Long Forward Payoff at time $T$',
                  'Short Forward Payoff at time $T$'])

# %% [block 2] the same call with the new plotter
from plotter import my_plotter          # the new plotter replaces cell 9

fig, ax = my_plotter(x_stock_price, [y_forward_payoff, y_forward_payoff__short],
                     layout=layout,
                     names=['Long Forward Payoff at time $T$',
                            'Short Forward Payoff at time $T$'])

# %% [block 3] two plots side by side in one figure, saved to a file
x = np.arange(0, 200, 0.001)
call_payoff = np.maximum(x - 100, 0.0)
put_payoff = np.maximum(100 - x, 0.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
my_plotter(x, [call_payoff, put_payoff], ax=axes[0], names=["call", "put"],
           layout={"title": "Payoffs", "colors": ["tab:blue", "tab:red"]})
my_plotter(x, put_payoff - 8, ax=axes[1], layout={"title": "Put net profit",
           "hline": None, "vlines": [92], "grid": True})
fig.tight_layout()
fig.savefig("figures/hw21_subplots.pdf")
plt.show()

# %% [block 4] payoffs and net profits with options of every group
call_price, put_price = 5.0, 8.0        # premiums of cells 19 and 25
example_layout = {
    # text: title, axis labels, sizes and paddings
    "title": "Long call and long put, K = 100", "title_fontsize": 13,
    "x_label": "$S_T$", "y_label": "value at $T$", "label_fontsize": 12,
    "title_pad": 12, "label_pad": 8, "tick_fontsize": 10,
    # curves: one entry per curve (a list is cycled over the curves)
    "colors": ["tab:blue", "tab:blue", "tab:red", "tab:red"],
    "linestyles": ["-", "--", "-", "--"], "linewidth": 2.2,
    # axes: reference lines, limits, grid and legend
    "hline": None, "vlines": [92, 105], "ylim": (-15, 100), "grid": True,
    "legend_loc": "upper center", "legend_fontsize": 9,
    # output: figure size and file
    "figsize": (7.5, 3.8), "save_path": "figures/hw21_payoffs.pdf",
}
fig, ax = my_plotter(x, [call_payoff, call_payoff - call_price,
                         put_payoff, put_payoff - put_price],
                     layout=example_layout,
                     names=["call payoff", "call net profit (c = 5)",
                            "put payoff", "put net profit (p = 8)"])

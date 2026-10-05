"""HW 2/1 -- a plotter whose layout is fully customizable.

Drop-in replacement for the notebook's ``my_plotter`` (Lesson 3, cell 9):
the old call ``my_plotter(x, y, layout={...}, names=[...])`` still works and,
without layout options, draws the figure of cell 9 (its values are defaults).

Changes with respect to the notebook version
  * every layout setting is a key of DEFAULT_LAYOUT and can be overridden;
  * unknown layout keys raise an error (typos do not pass silently);
  * no mutable default argument (layout=None instead of layout={});
  * one curve or many curves are detected with np.ndim, so integer and
    NumPy inputs work (the notebook test only accepted Python floats);
  * title/label paddings are passed to the axes, not to plt.rcParams,
    so the current figure is affected and later figures are not;
  * new options: y-label, colours, line styles, markers, opacity, limits,
    log axes, grid, reference lines, legend position, saving to file;
  * the figure can be drawn into an existing axes (for subplots);
  * returns (fig, ax) so the caller can keep editing the figure.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LAYOUT: Dict[str, Any] = {
    "figsize": (8, 4),        # inches (width, height)              [cell 9]
    "title": None,            # title of the axes
    "x_label": None,          # x-axis label
    "y_label": None,          # y-axis label                        (new)
    "title_fontsize": 20,     #                                     [cell 9]
    "label_fontsize": 16,     # x- and y-label                      [cell 9]
    "legend_fontsize": 16,    #                                     [cell 9]
    "tick_fontsize": None,    # None: Matplotlib default            [cell 9]
    "title_pad": 30,          # points between title and axes       [cell 9]
    "label_pad": 20,          # points between label and axis       [cell 9]
    "linewidth": 4,           #                                     [cell 9]
    "colors": None,           # colour or list, cycled over curves  (new)
    "linestyles": "solid",    # style or list like ["-", "--"]      [cell 9]
    "markers": None,          # marker or list like [None, "o"]     (new)
    "alpha": 1.0,             # opacity of the curves               (new)
    "legend_loc": "best",     # position of the legend              (new)
    "grid": False,            # True: light dashed grid             (new)
    "hline": 0.0,             # y of the dashed black line, or None [cell 9]
    "vlines": None,           # x values of dotted vertical lines   (new)
    "xlim": None,             # (xmin, xmax)                        (new)
    "ylim": None,             # (ymin, ymax)                        (new)
    "logx": False,            # logarithmic x-axis                  (new)
    "logy": False,            # logarithmic y-axis                  (new)
    "save_path": None,        # e.g. "figures/plot.pdf"             (new)
    "show": None,             # None: show only a figure created here
}


def _as_curves(x: np.ndarray, y) -> list:
    """Return y as a list of 1-D float arrays with the same length as x."""
    if np.ndim(y) == 1:
        curves = [np.asarray(y, dtype=float)]
    else:
        curves = [np.asarray(item, dtype=float) for item in y]
    for i, curve in enumerate(curves):
        if curve.shape != x.shape:
            raise ValueError(
                f"curve {i} has shape {curve.shape} but x has shape {x.shape}")
    return curves


def my_plotter(x: Sequence[float], y, layout: Optional[Dict[str, Any]] = None,
               names: Optional[Sequence[str]] = None, ax=None):
    """Plot one curve (1-D ``y``) or several curves (list / 2-D ``y``).

    ``layout`` overrides any key of DEFAULT_LAYOUT.  ``names`` gives one
    legend entry per curve.  ``ax`` lets you draw into an existing axes.
    Returns ``(fig, ax)``.
    """
    opts = dict(DEFAULT_LAYOUT)
    for key, value in (layout or {}).items():
        if key not in DEFAULT_LAYOUT:
            raise KeyError(f"unknown layout key {key!r}; "
                           f"allowed keys: {sorted(DEFAULT_LAYOUT)}")
        opts[key] = value

    x = np.asarray(x, dtype=float)
    curves = _as_curves(x, y)
    if names is not None and len(names) != len(curves):
        raise ValueError(f"{len(names)} names given for {len(curves)} curves")

    created = ax is None                    # False: draw into a given axes
    if created:
        ax = plt.subplots(figsize=opts["figsize"])[1]
    fig = ax.figure

    for i, curve in enumerate(curves):
        kwargs = {"linewidth": opts["linewidth"], "alpha": opts["alpha"]}
        for key, option in (("color", "colors"), ("linestyle", "linestyles"),
                            ("marker", "markers")):
            values = opts[option]
            if isinstance(values, str):         # "red" means ["red"]
                values = [values]
            if values is not None:
                kwargs[key] = values[i % len(values)]
        if names is not None:
            kwargs["label"] = names[i]
        ax.plot(x, curve, **kwargs)

    if opts["hline"] is not None:
        ax.axhline(opts["hline"], linestyle="--", color="black", linewidth=1)
    for x_value in opts["vlines"] or []:
        ax.axvline(x_value, linestyle=":", color="gray", linewidth=1)
    if opts["title"]:
        ax.set_title(opts["title"], fontsize=opts["title_fontsize"],
                     pad=opts["title_pad"])
    if opts["x_label"]:
        ax.set_xlabel(opts["x_label"], fontsize=opts["label_fontsize"],
                      labelpad=opts["label_pad"])
    if opts["y_label"]:
        ax.set_ylabel(opts["y_label"], fontsize=opts["label_fontsize"],
                      labelpad=opts["label_pad"])
    if opts["xlim"] is not None:
        ax.set_xlim(*opts["xlim"])
    if opts["ylim"] is not None:
        ax.set_ylim(*opts["ylim"])
    if opts["logx"]:
        ax.set_xscale("log")
    if opts["logy"]:
        ax.set_yscale("log")
    if opts["tick_fontsize"] is not None:
        ax.tick_params(labelsize=opts["tick_fontsize"])
    if opts["grid"]:
        ax.grid(True, alpha=0.3, linestyle="--")
    else:
        ax.grid(False)
    if names is not None:
        ax.legend(loc=opts["legend_loc"], fontsize=opts["legend_fontsize"])
    if created:                             # keep the layout of a user figure
        fig.tight_layout()
    if opts["save_path"]:
        fig.savefig(opts["save_path"])
    show = created if opts["show"] is None else opts["show"]
    if show:
        plt.show()
    return fig, ax

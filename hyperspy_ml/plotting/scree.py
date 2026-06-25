# -*- coding: utf-8 -*-
"""Scree plot engine for DecompositionResult."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from hyperspy.misc import utils as hs_utils
from matplotlib.ticker import FuncFormatter, MaxNLocator

from hyperspy_ml.plotting._helpers import _make_signal_from_array


def plot_scree(
    explained_variance_ratio,
    centre=None,
    algorithm="Decomposition",
    n=30,
    log=True,
    threshold=0,
    hline="auto",
    vline=False,
    xaxis_type="index",
    xaxis_labeling=None,
    signal_fmt=None,
    noise_fmt=None,
    fig=None,
    ax=None,
    number_significant_components=None,
    **figure_kwargs,
):
    """Core scree plot engine — used by DecompositionResult.plot_scree()."""
    s = _make_signal_from_array(explained_variance_ratio)

    n_max = len(explained_variance_ratio)
    if n is None:
        n = n_max
    elif n > n_max:
        n = n_max

    if isinstance(threshold, float):
        if not 0 < threshold < 1:
            raise ValueError("Variance threshold should be between 0 and 1")
        if threshold < s.data.min():
            n_signal_pcs = n
        else:
            n_signal_pcs = np.where((s < threshold).data)[0][0]
    else:
        n_signal_pcs = threshold
        if n_signal_pcs == 0:
            hline = False

    if vline:
        if number_significant_components is None:
            vline = False
        else:
            idx_nsc = number_significant_components - 1

    cutoff = None
    if hline == "auto":
        if isinstance(threshold, float):
            cutoff = threshold
        else:
            hline = False
    elif hline:
        if isinstance(threshold, float):
            cutoff = threshold
        elif n_signal_pcs > 0:
            cutoff = s.data[n_signal_pcs - 1]
    else:
        hline = False

    if signal_fmt is None:
        signal_fmt = {
            "c": "#C24D52",
            "linestyle": "",
            "marker": "^",
            "markersize": 10,
            "zorder": 3,
        }
    if noise_fmt is None:
        noise_fmt = {
            "c": "#4A70B0",
            "linestyle": "",
            "marker": "o",
            "markersize": 10,
            "zorder": 3,
        }

    if xaxis_labeling is None:
        xaxis_labeling = "cardinal" if xaxis_type == "index" else "ordinal"

    is_centred = centre is not None
    axes_titles = {
        "y": "Explained variance ratio"
        if is_centred
        else "Proportion of total variation",
        "x": f"{'Principal component' if is_centred else 'Component'} {xaxis_type}",
    }

    if n < s.axes_manager[-1].size:
        s = s.isig[:n]

    if fig is None:
        fig = plt.figure(**figure_kwargs)
    if ax is None:
        ax = fig.add_subplot(111)

    if log:
        ax.set_yscale("log")

    if hline and cutoff is not None:
        ax.axhline(cutoff, linewidth=2, color="gray", linestyle="dashed", zorder=1)

    if vline:
        ax.axvline(idx_nsc, linewidth=2, color="gray", linestyle="dashed", zorder=1)

    index_offset = 0 if xaxis_type == "index" else 1

    if n_signal_pcs == n:
        ax.plot(range(index_offset, index_offset + n), s.isig[:n].data, **signal_fmt)
    elif n_signal_pcs > 0:
        ax.plot(
            range(index_offset, index_offset + n_signal_pcs),
            s.isig[:n_signal_pcs].data,
            **signal_fmt,
        )
        ax.plot(
            range(index_offset + n_signal_pcs, index_offset + n),
            s.isig[n_signal_pcs:n].data,
            **noise_fmt,
        )
    else:
        ax.plot(range(index_offset, index_offset + n), s.isig[:n].data, **noise_fmt)

    if xaxis_labeling == "cardinal":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: hs_utils.ordinal(x)))

    ax.set_ylabel(axes_titles["y"])
    ax.set_xlabel(axes_titles["x"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.margins(0.05)
    ax.autoscale()
    ax.set_title(f"{algorithm} Scree Plot", y=1.01)
    return ax

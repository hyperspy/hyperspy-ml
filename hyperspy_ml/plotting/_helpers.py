# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Plotting helpers extracted from MVATools in signal.py."""

from __future__ import annotations

import numpy as np
from hyperspy.signals import Signal1D, Signal2D


def _make_signal_from_array(arr, source_signal=None):
    """Wrap a numpy array as a Signal1D or Signal2D for plotting.

    Uses the source signal's axes for calibration if available.
    """
    arr = np.asarray(arr, dtype=float)
    ndim = arr.ndim
    if ndim == 1:
        s = Signal1D(arr)
    elif ndim == 2:
        s = Signal2D(arr)
    else:
        s = Signal1D(arr)

    if source_signal is not None:
        try:
            sig_axes = source_signal.axes_manager.signal_axes
            if ndim == 1:
                s.axes_manager[0].scale = sig_axes[0].scale
                s.axes_manager[0].offset = sig_axes[0].offset
                s.axes_manager[0].units = sig_axes[0].units
                s.axes_manager[0].name = sig_axes[0].name
        except Exception:
            pass
    return s


def _get_components(result, idx=None):
    """Return components as a 1D numpy array (optionally indexed)."""
    c = np.asarray(result.components, dtype=float)
    if idx is not None:
        return c[:, idx]
    return c


def _get_scores(result, idx=None):
    """Return scores as a 1D numpy array (optionally indexed)."""
    s = np.asarray(result.scores, dtype=float)
    if idx is not None:
        return s[:, idx]
    return s


def _get_bss_components(result, idx=None):
    """Return BSS components."""
    if result.bss_components is None:
        raise ValueError("No BSS results available")
    c = np.asarray(result.bss_components, dtype=float)
    if idx is not None:
        return c[:, idx]
    return c


def _get_bss_scores(result, idx=None):
    """Return BSS scores."""
    if result.bss_scores is None:
        raise ValueError("No BSS results available")
    s = np.asarray(result.bss_scores, dtype=float)
    if idx is not None:
        return s[:, idx]
    return s


def _plot_component(ax, component, title="", **kwargs):
    """Plot a single component on *ax*."""
    ax.plot(component, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Channel")
    return ax


def _plot_score(ax, score, shape=None, title="", **kwargs):
    """Plot scores (can be 1D or 2D navigation)."""
    if shape is not None and len(shape) > 1:
        s2d = score.reshape(shape)
        ax.imshow(s2d, aspect="auto", **kwargs)
    else:
        ax.plot(score, **kwargs)
    ax.set_title(title)
    return ax


def _plot_cluster_signals(ax, labels, signals, title=""):
    """Plot cluster signals as overlaid lines with legend."""
    n_clusters = labels.shape[0]
    for i in range(n_clusters):
        ax.plot(signals[i], label=f"Cluster {i + 1}")
    ax.legend()
    ax.set_title(title)
    return ax


def _plot_cluster_labels(ax, labels, shape, title=""):
    """Plot cluster labels as a 2D image."""
    n_pixels = labels.shape[1]
    cluster_map = np.zeros(n_pixels, dtype=float)
    for i in range(labels.shape[0]):
        cluster_map[labels[i]] = i + 1
    if shape is not None:
        cluster_map = cluster_map.reshape(shape)
    ax.imshow(cluster_map, aspect="auto", cmap="tab10")
    ax.set_title(title)
    return ax


def _plot_cluster_distances(ax, distances, shape=None, title=""):
    """Plot cluster distances."""
    for i in range(distances.shape[0]):
        d = distances[i]
        if shape is not None:
            d = d.reshape(shape)
        ax.plot(d.flatten() if shape else d, label=f"Cluster {i + 1}")
    ax.legend()
    ax.set_title(title)
    return ax

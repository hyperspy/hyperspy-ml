# -*- coding: utf-8 -*-
"""Component/score plot engine."""

from __future__ import annotations

import matplotlib.pyplot as plt

from hyperspy_ml.plotting._helpers import (
    _get_components,
    _get_scores,
    _plot_component,
    _plot_score,
)


def plot_decomposition_components(result, source_signal=None, **kwargs):
    """Plot decomposition components from a DecompositionResult."""
    c = _get_components(result)
    n = c.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), squeeze=False)
    for i in range(n):
        _plot_component(axes[i, 0], c[:, i], title=f"Component {i + 1}", **kwargs)
    fig.tight_layout()
    return axes


def plot_decomposition_scores(result, nav_shape=None, source_signal=None, **kwargs):
    """Plot decomposition scores from a DecompositionResult."""
    s = _get_scores(result)
    n = s.shape[1]
    shape = nav_shape or (s.shape[0],)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), squeeze=False)
    for i in range(n):
        _plot_score(axes[i, 0], s[:, i], shape=shape, title=f"Score {i + 1}", **kwargs)
    fig.tight_layout()
    return axes


def plot_bss_components(result, source_signal=None, **kwargs):
    """Plot BSS components from a BSSResult."""
    from hyperspy_ml.plotting._helpers import _get_bss_components

    c = _get_bss_components(result)
    n = c.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), squeeze=False)
    for i in range(n):
        _plot_component(axes[i, 0], c[:, i], title=f"IC {i + 1}", **kwargs)
    fig.tight_layout()
    return axes


def plot_bss_scores(result, nav_shape=None, source_signal=None, **kwargs):
    """Plot BSS scores from a BSSResult."""
    from hyperspy_ml.plotting._helpers import _get_bss_scores

    s = _get_bss_scores(result)
    n = s.shape[1]
    shape = nav_shape or (s.shape[0],)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2 * n), squeeze=False)
    for i in range(n):
        _plot_score(axes[i, 0], s[:, i], shape=shape, title=f"IC {i + 1}", **kwargs)
    fig.tight_layout()
    return axes

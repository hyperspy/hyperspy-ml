# -*- coding: utf-8 -*-
"""Cluster plot engine."""

from __future__ import annotations

import matplotlib.pyplot as plt

from hyperspy_ml.plotting._helpers import (
    _plot_cluster_distances,
    _plot_cluster_labels,
    _plot_cluster_signals,
)


def plot_cluster_signals(result, **kwargs):
    """Plot cluster sum/centroid signals."""
    fig, ax = plt.subplots()
    _plot_cluster_signals(
        ax, result.cluster_labels, result.cluster_sum_signals, title="Cluster Signals"
    )
    fig.tight_layout()
    return ax


def plot_cluster_labels(result, nav_shape=None, **kwargs):
    """Plot cluster labels as spatial map."""
    fig, ax = plt.subplots()
    _plot_cluster_labels(ax, result.cluster_labels, nav_shape, title="Cluster Labels")
    return ax


def plot_cluster_distances(result, nav_shape=None, **kwargs):
    """Plot cluster distances."""
    fig, ax = plt.subplots()
    _plot_cluster_distances(
        ax, result.cluster_distances, nav_shape, title="Cluster Distances"
    )
    return ax

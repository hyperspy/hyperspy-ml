# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Tests for the Clustering stage (Task 5).

ALL-DIFFERENT dimensions (7, 11, 13).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import ClusterResult  # noqa: E402
from hyperspy_ml.stages.clustering import Clustering  # noqa: E402
from hyperspy_ml.stages.decomposition import Decomposition  # noqa: E402

skip_sklearn = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)


def _make_result_and_signal(rng_seed=42, rank=5):
    """DecompositionResult + Signal1D with all-diff dims (7, 11, 13)."""
    rng = np.random.default_rng(rng_seed)
    nav, sig = 77, 13
    U = rng.standard_normal((nav, rank))
    V = rng.standard_normal((sig, rank))
    X = U @ V.T
    s = Signal1D(X.reshape(7, 11, sig))
    stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
    result = stage.fit_transform(s)
    result._nav_shape = s.axes_manager.navigation_shape[::-1]
    return result, s


@skip_sklearn
class TestKMeansClustering:
    def test_kmeans_basic(self):
        """KMeans clustering produces labels and centroids."""
        lr, s = _make_result_and_signal()
        clust = Clustering(
            n_clusters=3, cluster_source="decomposition", print_info=False
        )
        result = clust.fit_transform(lr)
        assert result.number_of_clusters == 3
        assert result.cluster_labels.shape == (3, 77)
        assert result.cluster_centroids.shape == (3, 5)  # n_comp features
        assert result.cluster_distances.shape == (3, 77)

    def test_cluster_labels_boolean(self):
        """cluster_labels is a boolean matrix."""
        lr, s = _make_result_and_signal()
        clust = Clustering(n_clusters=3, print_info=False)
        result = clust.fit_transform(lr)
        assert result.cluster_labels.dtype == bool

    def test_labels_sorted_by_size(self):
        """Clusters are sorted by size (descending)."""
        lr, s = _make_result_and_signal()
        clust = Clustering(n_clusters=3, print_info=False)
        result = clust.fit_transform(lr)
        sizes = result.cluster_labels.sum(axis=1)
        assert sizes[0] >= sizes[1] >= sizes[2]

    def test_no_n_clusters_returns_empty(self):
        """n_clusters=None returns empty result."""
        lr, s = _make_result_and_signal()
        clust = Clustering(n_clusters=None, print_info=False)
        result = clust.fit_transform(lr)
        assert result.cluster_labels is None
        assert result.number_of_clusters == 0


@skip_sklearn
class TestMiniBatchKMeans:
    def test_minibatchkmeans(self):
        """MiniBatchKMeans with partial_fit through clustering."""
        lr, s = _make_result_and_signal()
        clust = Clustering(n_clusters=3, algorithm="minibatchkmeans", print_info=False)
        result = clust.fit_transform(lr)
        assert result.number_of_clusters == 3
        assert result.cluster_labels is not None


@skip_sklearn
class TestGapStatistic:
    def test_gap_statistic_returns_best_k(self):
        """Gap statistic produces a best_k estimate."""
        lr, s = _make_result_and_signal()
        scaled_data = lr.scores[:, :3]

        best_k, curve = Clustering.estimate_number_of_clusters(
            scaled_data,
            max_clusters=5,
            algorithm="kmeans",
            metric="gap",
            n_ref=2,
            show_progressbar=False,
        )
        assert isinstance(best_k, int)
        assert best_k >= 2
        assert curve is not None
        assert len(curve) == 5  # max_clusters values

    def test_elbow_metric(self):
        """Elbow metric returns a best_k."""
        lr, s = _make_result_and_signal()
        scaled_data = lr.scores[:, :3]
        best_k, curve = Clustering.estimate_number_of_clusters(
            scaled_data,
            max_clusters=5,
            algorithm="kmeans",
            metric="elbow",
            show_progressbar=False,
        )
        assert isinstance(best_k, np.integer)

    def test_silhouette_metric(self):
        """Silhouette metric returns best_k list or int."""
        lr, s = _make_result_and_signal()
        scaled_data = lr.scores[:, :3]
        best_k, curve = Clustering.estimate_number_of_clusters(
            scaled_data,
            max_clusters=5,
            algorithm="kmeans",
            metric="silhouette",
            show_progressbar=False,
        )
        assert best_k is not None  # list or empty


@skip_sklearn
class TestPlotClusterMetric:
    def test_plot_renders(self):
        """plot_cluster_metric renders without error."""
        import matplotlib

        matplotlib.use("agg")
        lr, s = _make_result_and_signal()
        scaled_data = lr.scores[:, :3]

        best_k, curve = Clustering.estimate_number_of_clusters(
            scaled_data,
            max_clusters=4,
            algorithm="kmeans",
            metric="gap",
            n_ref=2,
            show_progressbar=False,
        )

        result = ClusterResult(
            cluster_metric="gap",
            cluster_metric_index=np.arange(1, 5),
            cluster_metric_data=np.asarray(curve),
        )
        ax = Clustering.plot_cluster_metric(result)
        assert ax is not None

    def test_plot_without_metric_raises(self):
        """plot_cluster_metric without metric data raises."""
        result = ClusterResult()
        with pytest.raises(ValueError, match="No metric data"):
            Clustering.plot_cluster_metric(result)


class TestClusteringValidation:
    @skip_sklearn
    def test_max_clusters_too_small(self):
        """max_clusters < 2 raises ValueError."""
        lr, s = _make_result_and_signal()
        with pytest.raises(ValueError, match="max_clusters must be"):
            Clustering.estimate_number_of_clusters(
                lr.scores[:, :3],
                max_clusters=1,
                algorithm="kmeans",
                show_progressbar=False,
            )

    @skip_sklearn
    def test_invalid_metric_raises(self):
        """Invalid metric raises ValueError."""
        lr, s = _make_result_and_signal()
        with pytest.raises(ValueError, match="metric must be"):
            Clustering.estimate_number_of_clusters(
                lr.scores[:, :3],
                max_clusters=3,
                algorithm="kmeans",
                metric="invalid",
                show_progressbar=False,
            )

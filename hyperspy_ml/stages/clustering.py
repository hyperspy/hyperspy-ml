# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy ML.
"""Clustering analysis stage with gap-statistic estimation.

Ported from :meth:`MVA.cluster_analysis` and
:meth:`MVA.estimate_number_of_clusters` (~1000+ lines).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
from hyperspy.external.progressbar import progressbar
from hyperspy.misc import utils as hs_utils

from hyperspy_ml.results.base import ClusterResult
from hyperspy_ml.utils.preprocessing import (
    _get_sklearn_clustering_algorithms,
    _get_sklearn_preprocessing_algorithms,
    cluster_algorithms,
    estimate_elbow_position,
    preprocessing_algorithms,
)

_logger = logging.getLogger(__name__)

SKLEARN_INSTALLED = importlib.util.find_spec("sklearn") is not None


def _check_sklearn():
    if not SKLEARN_INSTALLED:
        raise ImportError("clustering requires scikit-learn")


class Clustering:
    """Standalone clustering stage.

    Parameters
    ----------
    n_clusters : int or None
        Number of clusters.  If ``None``, no clustering is performed
        (useful with ``estimate_number_of_clusters``).
    cluster_source : str or BaseSignal
        Source data for clustering.
    source_for_centers : str or BaseSignal
        Source for reconstructing cluster-centre signals.
    preprocessing : str or object, default None
        ``"norm"``, ``"standard"``, ``"minmax"``, ``None``, or a
        sklearn-like preprocessor.
    preprocessing_kwargs : dict or None
    number_of_components : int or None
        Number of decompositions components to use.
    navigation_mask : ndarray or None
    signal_mask : ndarray or None
    algorithm : str or object, default None (KMeans)
        ``"kmeans"``, ``"agglomerative"``, ``"minibatchkmeans"``,
        ``"spectralclustering"``, or a custom object.
    print_info : bool, default True
    **kwargs
        Passed to the clustering algorithm.
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        cluster_source: str | Any = "decomposition",
        source_for_centers: str | Any | None = None,
        preprocessing: str | object | None = None,
        preprocessing_kwargs: dict[str, Any] | None = None,
        number_of_components: int | None = None,
        navigation_mask: Any = None,
        signal_mask: Any = None,
        algorithm: str | object | None = None,
        print_info: bool = True,
        **kwargs: Any,
    ) -> None:
        self.n_clusters = n_clusters
        self.cluster_source = cluster_source
        self.source_for_centers = source_for_centers
        self.preprocessing = preprocessing
        self.preprocessing_kwargs = preprocessing_kwargs or {}
        self.number_of_components = number_of_components
        self.navigation_mask = navigation_mask
        self.signal_mask = signal_mask
        self.algorithm = algorithm
        self.print_info = print_info
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, decomposition_result, signal=None):
        """Run clustering on a source.

        Parameters
        ----------
        decomposition_result : DecompositionResult
        signal : BaseSignal, optional
            Original signal (needed when ``cluster_source='signal'``).

        Returns
        -------
        ClusterResult
        """
        return self._run(decomposition_result, signal)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run(self, lr, signal):
        _check_sklearn()

        source = self.cluster_source
        src_centers = self.source_for_centers

        n_clusters = self.n_clusters
        n_comp = self.number_of_components
        nav_mask = self.navigation_mask

        # Determine number_of_components
        if n_comp is None:
            # Use all components by default
            n_comp = (
                lr.n_components
                if lr.n_components > 0
                else (lr.components.shape[1] if lr.components is not None else None)
            )

        # Get cluster data
        scaled_data = self._scale_data_for_clustering(source, lr, signal, n_comp)

        if n_clusters is None:
            return ClusterResult(params=self._build_params())

        # Run clustering
        cluster_algo = self._get_cluster_algorithm(
            self.algorithm, n_clusters=n_clusters, **self._extra_kwargs
        )
        alg = self._cluster_analysis(scaled_data, cluster_algo)
        labels = alg.labels_

        # Sort clusters by size (descending)
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_order = unique_labels[np.argsort(-counts)]
        remapped = np.zeros_like(labels)
        for new_idx, old_label in enumerate(sorted_order):
            remapped[labels == old_label] = new_idx
        labels = remapped

        # Build boolean cluster matrix
        cluster_labels = np.zeros((n_clusters, scaled_data.shape[0]), dtype=bool)
        for i in range(n_clusters):
            cluster_labels[i, labels == i] = True

        # Compute centroids and distances
        centroids = []
        distances = np.full((n_clusters, scaled_data.shape[0]), np.nan, dtype=float)
        cluster_centroid_signals = []
        cluster_sum_signals = []

        if nav_mask is not None:
            nav_mask_flat = _to_flat_bool(nav_mask)
        else:
            nav_mask_flat = np.zeros(scaled_data.shape[0], dtype=bool)

        if isinstance(src_centers, str) and src_centers in (
            "decomposition",
            "bss",
        ):
            loadings = lr.scores[:, :n_comp]
            factors = lr.components[:, :n_comp]
            for i in range(n_clusters):
                s_ld = loadings[cluster_labels[i, :], :].sum(0, keepdims=True)
                cluster_sum_signals.append((s_ld @ factors.T).squeeze())
                cdata = scaled_data[cluster_labels[i, :][~nav_mask_flat], :]
                centroid = cdata.mean(0)
                centroids.append(centroid)
                cdist = np.linalg.norm(scaled_data - centroid[np.newaxis, :], axis=1)
                m_ld = loadings[~nav_mask_flat, ...][np.argmin(cdist), ...]
                cluster_centroid_signals.append((m_ld @ factors.T).squeeze())
                distances[i, ~nav_mask_flat] = cdist
        else:
            cluster_data = scaled_data
            for i in range(n_clusters):
                cluster_sum_signals.append(
                    np.sum(cluster_data[cluster_labels[i, :], ...], axis=0)
                )
                cdata = scaled_data[cluster_labels[i, :][~nav_mask_flat], :]
                centroid = cdata.mean(0)
                centroids.append(centroid)
                cdist = np.linalg.norm(scaled_data - centroid[np.newaxis, :], axis=1)
                csig = cluster_data[~nav_mask_flat, ...][np.argmin(cdist), ...]
                cluster_centroid_signals.append(csig)
                distances[i, ~nav_mask_flat] = cdist

        result = ClusterResult(
            cluster_labels=cluster_labels,
            cluster_centroids=np.asarray(centroids),
            cluster_distances=distances,
            cluster_sum_signals=np.stack(cluster_sum_signals),
            cluster_centroid_signals=np.stack(cluster_centroid_signals),
            number_of_clusters=n_clusters,
            cluster_algorithm=str(self.algorithm or "KMeans"),
            params=self._build_params(),
        )
        return result

    # ------------------------------------------------------------------
    # Data scaling
    # ------------------------------------------------------------------

    def _scale_data_for_clustering(self, source, lr, signal, n_comp):
        """Extract and preprocess the source data for clustering."""
        data = self._get_cluster_signal(source, lr, signal, n_comp)
        preprocessor = self._get_cluster_preprocessing_algorithm(
            self.preprocessing, **self.preprocessing_kwargs
        )
        if preprocessor is not None:
            data = preprocessor.fit_transform(data)
        return data

    def _get_cluster_signal(self, source, lr, signal, n_comp):
        """Retrieve the data to cluster."""
        _check_sklearn()
        if isinstance(source, str):
            if source == "decomposition":
                return lr.scores[:, :n_comp]
            elif source == "bss":
                return lr.bss_scores[:, :n_comp]
            elif source == "signal":
                if signal is None:
                    raise ValueError("signal= required for cluster_source='signal'")
                return np.asarray(
                    signal.data.reshape(
                        signal.axes_manager.navigation_size,
                        signal.axes_manager.signal_size,
                    )
                )
            else:
                raise ValueError(f"Unknown cluster_source: {source!r}")
        elif hs_utils.is_hyperspy_signal(source):
            return np.asarray(source.data)
        return np.asarray(source)

    # ------------------------------------------------------------------
    # Algorithm factories
    # ------------------------------------------------------------------

    def _get_cluster_algorithm(self, algorithm, **kwargs):
        """Instantiate the sklearn clusterer."""
        _check_sklearn()
        if isinstance(algorithm, str) and algorithm in cluster_algorithms:
            return _get_sklearn_clustering_algorithms(algorithm)(**kwargs)
        elif algorithm is None:
            return _get_sklearn_clustering_algorithms(None)(**kwargs)
        elif hasattr(algorithm, "fit"):
            return algorithm
        raise ValueError(
            f"Clustering algorithm must be one of "
            f"{list(cluster_algorithms.keys())} or a fit()-able object"
        )

    def _get_cluster_preprocessing_algorithm(self, algorithm, **kwargs):
        """Instantiate the sklearn preprocessor."""
        if algorithm is None:
            return None
        if algorithm in preprocessing_algorithms:
            _check_sklearn()
            return _get_sklearn_preprocessing_algorithms(algorithm)(**kwargs)
        if hasattr(algorithm, "fit_transform"):
            return algorithm
        raise ValueError(
            f"Preprocessing must be one of "
            f"{list(preprocessing_algorithms.keys())}"
            " or a fit_transform()-able object"
        )

    @staticmethod
    def _cluster_analysis(data, cluster_algo):
        """Run the clusterer."""
        if hasattr(cluster_algo, "labels_"):
            return cluster_algo
        if hasattr(cluster_algo, "fit_predict"):
            cluster_algo.fit_predict(data)
        elif hasattr(cluster_algo, "fit"):
            cluster_algo.fit(data)
        return cluster_algo

    # ------------------------------------------------------------------
    # Gap statistic and cluster-number estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _distances_within_cluster(data, memberships, squared=True, summed=False):
        """Compute within-cluster distances."""
        import sklearn

        distances = [
            sklearn.metrics.pairwise.euclidean_distances(
                data[memberships == c, :],
                squared=squared,
            )
            for c in range(int(np.max(memberships)) + 1)
        ]
        result = [np.mean(x, axis=0) / 2.0 for x in distances]
        if summed:
            result = [np.sum(x) for x in result]
        return result

    @staticmethod
    def estimate_number_of_clusters(
        scaled_data,
        max_clusters=10,
        algorithm=None,
        metric="gap",
        n_ref=4,
        show_progressbar=True,
        **kwargs,
    ):
        """Estimate optimal *k* via elbow, silhouette, or gap statistic.

        Parameters
        ----------
        scaled_data : ndarray, shape (n_samples, n_features)
        max_clusters : int
        algorithm : str or None
        metric : {'elbow', 'silhouette', 'gap'}
        n_ref : int
            Number of reference distributions for gap.
        show_progressbar : bool
        **kwargs : passed to clustering algorithm

        Returns
        -------
        best_k : int or list[int]
        metrics_curve : ndarray or None
        """
        import sklearn

        _check_sklearn()
        if max_clusters < 2:
            raise ValueError("max_clusters must be >= 2")
        if algorithm not in cluster_algorithms:
            raise ValueError(
                f"algorithm must be one of {list(cluster_algorithms.keys())}"
            )

        min_k = 2
        best_k = None
        metrics_curve = None

        if metric == "elbow":
            k_range = list(range(1, max_clusters + 1))
            inertia = np.zeros(len(k_range))
            with progressbar(
                total=len(k_range), disable=not show_progressbar, leave=True
            ) as pbar:
                for idx, k in enumerate(k_range):
                    ca = _get_sklearn_clustering_algorithms(algorithm)(
                        n_clusters=k, **kwargs
                    )
                    alg = Clustering._cluster_analysis(scaled_data, ca)
                    D = Clustering._distances_within_cluster(
                        scaled_data, alg.labels_, summed=True
                    )
                    W = np.sum(D)
                    inertia[idx] = np.log(W)
                    pbar.update(1)
            metrics_curve = inertia
            best_k = estimate_elbow_position(metrics_curve, log=False) + min_k

        elif metric == "silhouette":
            k_range = list(range(2, max_clusters + 1))
            scores = []
            with progressbar(
                total=len(k_range), disable=not show_progressbar, leave=True
            ) as pbar:
                for k in k_range:
                    ca = _get_sklearn_clustering_algorithms(algorithm)(
                        n_clusters=k, **kwargs
                    )
                    alg = Clustering._cluster_analysis(scaled_data, ca)
                    score = sklearn.metrics.silhouette_score(scaled_data, alg.labels_)
                    scores.append(score)
                    pbar.update(1)
            metrics_curve = np.asarray(scores)
            best_k = []
            max_val = -1.0
            for u in range(len(scores) - 1):
                if scores[u] > scores[u - 1] and scores[u] > scores[u + 1]:
                    best_k.append(u + min_k)
                    max_val = max(scores[u], max_val)
            if scores[0] > max_val:
                best_k.insert(0, min_k)

        elif metric == "gap":
            k_range = list(range(1, max_clusters + 1))
            ref_inertia = np.zeros(len(k_range))
            ref_std = np.zeros(len(k_range))
            data_inertia = np.zeros(len(k_range))
            ref = np.zeros(scaled_data.data.shape)
            local_inertia = np.zeros(n_ref)

            for f_idx in range(scaled_data.data.shape[1]):
                xmin = np.min(scaled_data[:, f_idx])
                xmax = np.max(scaled_data[:, f_idx])
                ref[:, f_idx] = np.linspace(xmin, xmax, endpoint=True, num=ref.shape[0])

            with progressbar(
                total=n_ref * len(k_range),
                disable=not show_progressbar,
                leave=True,
            ) as pbar:
                for o_idx, k in enumerate(k_range):
                    kw = dict(**kwargs)
                    if algorithm == "kmeans":
                        kw["n_init"] = 1
                    ca = _get_sklearn_clustering_algorithms(algorithm)(
                        n_clusters=k, **kw
                    )
                    alg = Clustering._cluster_analysis(scaled_data, ca)
                    D = Clustering._distances_within_cluster(
                        scaled_data, alg.labels_, squared=True, summed=True
                    )
                    W = np.sum(D)
                    data_inertia[o_idx] = np.log(W)

                    for i_idx in range(n_ref):
                        ca2 = _get_sklearn_clustering_algorithms(algorithm)(
                            n_clusters=k, **kw
                        )
                        alg2 = Clustering._cluster_analysis(ref, ca2)
                        D2 = Clustering._distances_within_cluster(
                            ref, alg2.labels_, squared=True, summed=True
                        )
                        W2 = np.sum(D2)
                        local_inertia[i_idx] = np.log(W2)
                        pbar.update(1)
                    ref_inertia[o_idx] = np.mean(local_inertia)
                    ref_std[o_idx] = np.std(local_inertia)

            std_error = np.sqrt(1.0 + 1.0 / n_ref) * ref_std
            std_error = abs(std_error)
            gap = ref_inertia - data_inertia
            metrics_curve = gap

            best_k = min_k + 1
            for i in range(1, len(k_range) - 1):
                if i < len(k_range) - 1:
                    if gap[i] >= (gap[i + 1] - std_error[i + 1]):
                        best_k = i + min_k
                        break
                else:
                    if gap[i] > gap[i - 1] + std_error[i - 1]:
                        best_k = len(k_range)
        else:
            raise ValueError(
                f"metric must be 'elbow', 'silhouette', or 'gap', not {metric!r}"
            )

        return best_k, metrics_curve

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def plot_cluster_metric(result, fig=None, **kwargs):
        """Plot the cluster metric (elbow/silhouette/gap curve).

        Parameters
        ----------
        result : ClusterResult
        fig : matplotlib.figure.Figure or None
        **kwargs : passed to plt.figure

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if result.cluster_metric_index is None:
            raise ValueError("No metric data — run estimate_number_of_clusters first")

        if fig is None:
            fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111)

        ax.plot(
            result.cluster_metric_index,
            result.cluster_metric_data,
            "o-",
        )
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel(
            {
                "elbow": "Distance metric",
                "silhouette": "Silhouette score",
                "gap": "Gap statistic",
            }.get(result.cluster_metric, "Metric value")
        )
        ax.set_title(f"Cluster metric ({result.cluster_metric})")
        return ax

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "cluster_source": str(self.cluster_source),
            "preprocessing": str(self.preprocessing),
            "algorithm": str(self.algorithm),
            **self._extra_kwargs,
        }


def _to_flat_bool(mask):
    """Return a flat boolean numpy array from various mask types."""
    if mask is None:
        return None
    if hasattr(mask, "data"):
        mask = mask.data
    return np.asarray(mask, dtype=bool).ravel()

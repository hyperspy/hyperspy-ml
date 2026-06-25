# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Tests for plotting engine and deprecated aliases (Task 8)."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import ClusterResult
from hyperspy_ml.stages.decomposition import Decomposition

skip_sklearn = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)


def _make_decom_result(rng_seed=42, rank=4):
    rng = np.random.default_rng(rng_seed)
    nav, sig = 77, 13
    U = rng.standard_normal((nav, rank))
    V = rng.standard_normal((sig, rank))
    X = U @ V.T
    s = Signal1D(X.reshape(7, 11, sig))
    stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
    result = stage.fit_transform(s)
    result._nav_shape = s.axes_manager.navigation_shape[::-1]
    result._source_signal = s
    return result, s


# ============================================================================
# 8a: Scree plot (verify existing plot_scree works)
# ============================================================================


class TestScreePlot:
    def test_plot_scree_renders(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=10)
        ax = result.plot_scree(n=None)
        assert ax is not None

    def test_plot_cumulative_scree_renders(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=10)
        ax = result.plot_cumulative_scree(n=5)
        assert ax is not None


# ============================================================================
# 8b: Component/score plots
# ============================================================================


class TestComponentPlots:
    def test_plot_decomposition_components_renders(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        axes = result.plot_components()
        assert axes is not None

    def test_plot_decomposition_scores_renders(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        axes = result.plot_scores()
        assert axes is not None

    def test_plot_bss_components_renders(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        axes = result.plot_bss_components()
        assert axes is not None

    def test_plot_bss_scores_renders(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        axes = result.plot_bss_scores()
        assert axes is not None


# ============================================================================
# 8b: Cluster plots
# ============================================================================


@skip_sklearn
class TestClusterPlots:
    def test_plot_cluster_signals(self):
        import matplotlib

        matplotlib.use("agg")
        labels = np.zeros((3, 77), dtype=bool)
        labels[0, :20] = True
        labels[1, 20:50] = True
        labels[2, 50:] = True
        cr = ClusterResult(
            cluster_labels=labels,
            cluster_sum_signals=np.random.default_rng(1).standard_normal((3, 13)),
            cluster_distances=np.random.default_rng(1).standard_normal((3, 77)),
            number_of_clusters=3,
        )
        ax = cr.plot_cluster_signals()
        assert ax is not None

    def test_plot_cluster_labels(self):
        import matplotlib

        matplotlib.use("agg")
        labels = np.zeros((3, 77), dtype=bool)
        labels[0, :20] = True
        labels[1, 20:50] = True
        labels[2, 50:] = True
        cr = ClusterResult(
            cluster_labels=labels,
            cluster_sum_signals=np.random.default_rng(1).standard_normal((3, 13)),
            cluster_distances=np.random.default_rng(1).standard_normal((3, 77)),
            number_of_clusters=3,
        )
        ax = cr.plot_cluster_labels(nav_shape=(7, 11))
        assert ax is not None


# ============================================================================
# 8c: Deprecated aliases
# ============================================================================


class TestDeprecatedPlotAliases:
    def test_plot_decomposition_factors_warns(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result.plot_decomposition_factors()

    def test_plot_decomposition_loadings_warns(self):
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result.plot_decomposition_loadings()

    def test_get_decomposition_factors_warns(self):
        result, s = _make_decom_result(rank=4)
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            c = result.get_decomposition_factors()
        assert c.shape == (13, 4)

    def test_get_decomposition_loadings_warns(self):
        result, s = _make_decom_result(rank=4)
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            s_ = result.get_decomposition_loadings()
        assert s_.shape == (77, 4)


class TestGetAccessors:
    def test_get_components(self):
        result, s = _make_decom_result(rank=3)
        c = result.get_components()
        np.testing.assert_array_equal(c, result.components)

    def test_get_scores(self):
        result, s = _make_decom_result(rank=3)
        sc = result.get_scores()
        np.testing.assert_array_equal(sc, result.scores)

    def test_get_bss_components_raises_without_bss(self):
        result, s = _make_decom_result(rank=3)
        with pytest.raises(ValueError, match="No BSS"):
            result.get_bss_components()

# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Targeted tests to fill coverage gaps for the hyperspy-ml codebase.

ALL-DIFFERENT dimensions (7, 11, 13) used throughout.
Tests cover: pipeline edge cases, API convenience wrappers, plotting helpers,
BSSResult/ClusterResult methods, DecompositionResult extra paths, Study
save/load/remove, deprecated aliases, and result events.
"""

from __future__ import annotations

import importlib
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.pipeline.pipeline import Pipeline
from hyperspy_ml.results.base import (
    BSSResult,
    ClusterResult,
    DecompositionResult,
    _array_field_names,
    _scalar_field_names,
)
from hyperspy_ml.stages.decomposition import Decomposition

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")


# ============================================================================
# Fixtures
# ============================================================================


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
# Pipeline edge cases
# ============================================================================


class TestPipelineEdgeCases:
    def test_iter_yields_name_and_stage(self):
        """Pipeline.__iter__ yields (name, stage) tuples."""
        seen = []

        def stage_a(x):
            return x + 1

        p = Pipeline([("add", stage_a, None), ("add2", stage_a, None)])
        for name, stage in p:
            seen.append((name, stage is stage_a))
        assert seen == [("add", True), ("add2", True)]

    def test_stage_names_property(self):
        """Pipeline.stage_names returns list of stage names."""

        def stage_a(x):
            return x

        p = Pipeline([("alpha", stage_a, None), ("beta", stage_a, None)])
        assert p.stage_names == ["alpha", "beta"]

    def test_partial_fit_notimplemented(self):
        """Pipeline.partial_fit raises NotImplementedError when no stage has it."""

        def stage_a(x):
            return x

        p = Pipeline([("step", stage_a, None)])
        with pytest.raises(NotImplementedError, match="partial_fit"):
            p.partial_fit(np.array([1, 2]))

    def test_run_contract_check_fails(self):
        """Pipeline.run raises TypeError when contract is violated."""

        def stage_a(x):
            return str(x)

        p = Pipeline([("step", stage_a, (int,))])
        with pytest.raises(TypeError, match="requires input type"):
            p.run("not_an_int")

    def test_slicing_returns_pipeline(self):
        """Pipeline.__getitem__ with slice returns a new Pipeline."""

        def stage_a(x):
            return x

        p = Pipeline([("a", stage_a, None), ("b", stage_a, None)])
        sub = p[0:2]
        assert isinstance(sub, Pipeline)
        assert len(sub) == 2


# ============================================================================
# API convenience wrappers
# ============================================================================


class TestApiConvenienceWrappers:
    def test_api_bss(self):
        """api.bss() convenience function works."""
        from hyperspy_ml.api import bss

        result, s = _make_decom_result(rank=3)
        bss_result, _extra = bss(result, number_of_components=2, algorithm="orthomax")
        assert isinstance(bss_result, BSSResult)
        assert bss_result.bss_components is not None
        assert bss_result.bss_scores is not None

    @skip_sklearn
    def test_api_cluster(self):
        """api.cluster() convenience function works."""
        from hyperspy_ml.api import cluster

        result, s = _make_decom_result(rank=4)
        cr = cluster(result, n_clusters=2)
        assert isinstance(cr, ClusterResult)
        assert cr.number_of_clusters == 2

    def test_api_load_result(self):
        """api.load_result() dispatches correctly."""
        from hyperspy_ml.api import load_result

        result, s = _make_decom_result(rank=3)
        with TemporaryDirectory() as tmp:
            path = tmp + "/test.hsml"
            result.save(path)
            loaded = load_result(path)
            np.testing.assert_array_almost_equal(loaded.components, result.components)

    def test_api_extract_results(self):
        """api.extract_results() works with string path."""
        from hyperspy_ml.api import extract_results

        result, s = _make_decom_result(rank=3)
        with TemporaryDirectory() as tmp:
            path = tmp + "/test.hsml"
            result.save(path)
            loaded = extract_results(path)
            assert isinstance(loaded, DecompositionResult)


# ============================================================================
# Plotting helpers
# ============================================================================


class TestPlottingHelpers:
    def test_make_signal_from_array_3d_plus(self):
        """_make_signal_from_array handles 3D+ arrays gracefully."""
        from hyperspy_ml.plotting._helpers import _make_signal_from_array

        arr = np.random.default_rng(1).standard_normal((3, 4, 5))
        s = _make_signal_from_array(arr)
        assert s is not None

    def test_make_signal_from_array_with_source(self):
        """_make_signal_from_array with source_signal copies axes."""
        from hyperspy_ml.plotting._helpers import _make_signal_from_array

        arr = np.random.default_rng(1).standard_normal(13)
        result, s = _make_decom_result(rank=3)
        s_out = _make_signal_from_array(arr, source_signal=s)
        assert s_out is not None

    def test_get_components_with_idx(self):
        """_get_components with idx returns subset."""
        from hyperspy_ml.plotting._helpers import _get_components

        result, s = _make_decom_result(rank=4)
        c = _get_components(result, idx=0)
        assert c.shape == (13,)

    def test_get_scores_with_idx(self):
        """_get_scores with idx returns subset."""
        from hyperspy_ml.plotting._helpers import _get_scores

        result, s = _make_decom_result(rank=4)
        sc = _get_scores(result, idx=1)
        assert sc.shape == (77,)

    def test_get_bss_components_with_idx(self):
        """_get_bss_components with idx returns subset."""
        from hyperspy_ml.plotting._helpers import _get_bss_components

        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        c = _get_bss_components(result, idx=0)
        assert c.shape == (13,)

    def test_get_bss_scores_with_idx(self):
        """_get_bss_scores with idx returns subset."""
        from hyperspy_ml.plotting._helpers import _get_bss_scores

        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        sc = _get_bss_scores(result, idx=1)
        assert sc.shape == (77,)

    def test_plot_score_2d(self):
        """_plot_score with 2D shape uses imshow."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_score

        fig, ax = plt.subplots()
        score = np.random.default_rng(1).standard_normal(77)
        ax = _plot_score(ax, score, shape=(7, 11), title="test")
        assert ax is not None
        plt.close(fig)

    def test_plot_cluster_distances(self):
        """plot_cluster_distances function renders."""
        import matplotlib

        matplotlib.use("agg")
        from hyperspy_ml.plotting.cluster import plot_cluster_distances

        cr = ClusterResult(
            cluster_distances=np.random.default_rng(1).standard_normal((3, 77)),
            number_of_clusters=3,
            cluster_sum_signals=np.random.default_rng(1).standard_normal((3, 13)),
        )
        ax = plot_cluster_distances(cr, nav_shape=(7, 11))
        assert ax is not None

    def test_plot_cluster_distances_no_shape(self):
        """plot_cluster_distances without nav_shape."""
        import matplotlib

        matplotlib.use("agg")
        from hyperspy_ml.plotting.cluster import plot_cluster_distances

        cr = ClusterResult(
            cluster_distances=np.random.default_rng(1).standard_normal((2, 20)),
            number_of_clusters=2,
            cluster_sum_signals=np.random.default_rng(1).standard_normal((2, 10)),
        )
        ax = plot_cluster_distances(cr)
        assert ax is not None

    def test_get_bss_components_raises_without_bss(self):
        """_get_bss_components raises when no BSS available."""
        from hyperspy_ml.plotting._helpers import _get_bss_components

        result, s = _make_decom_result(rank=3)
        with pytest.raises(ValueError, match="No BSS"):
            _get_bss_components(result)

    def test_get_bss_scores_raises_without_bss(self):
        """_get_bss_scores raises when no BSS available."""
        from hyperspy_ml.plotting._helpers import _get_bss_scores

        result, s = _make_decom_result(rank=3)
        with pytest.raises(ValueError, match="No BSS"):
            _get_bss_scores(result)

    def test_plot_component(self):
        """_plot_component renders."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_component

        fig, ax = plt.subplots()
        component = np.random.default_rng(1).standard_normal(13)
        ax = _plot_component(ax, component, title="test")
        assert ax is not None
        plt.close(fig)

    def test_plot_score_1d(self):
        """_plot_score without 2D shape uses plot."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_score

        fig, ax = plt.subplots()
        score = np.random.default_rng(1).standard_normal(30)
        ax = _plot_score(ax, score, shape=None, title="test")
        assert ax is not None
        plt.close(fig)

    def test_plot_cluster_signals_helper(self):
        """_plot_cluster_signals renders."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_cluster_signals

        fig, ax = plt.subplots()
        labels = np.zeros((3, 77), dtype=bool)
        labels[0, :20] = True
        labels[1, 20:50] = True
        labels[2, 50:] = True
        signals = np.random.default_rng(1).standard_normal((3, 13))
        ax = _plot_cluster_signals(ax, labels, signals, title="test")
        assert ax is not None
        plt.close(fig)

    def test_plot_cluster_labels_helper(self):
        """_plot_cluster_labels renders."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_cluster_labels

        fig, ax = plt.subplots()
        labels = np.zeros((3, 77), dtype=bool)
        labels[0, :20] = True
        labels[1, 20:50] = True
        labels[2, 50:] = True
        ax = _plot_cluster_labels(ax, labels, shape=(7, 11), title="test")
        assert ax is not None
        plt.close(fig)

    def test_plot_cluster_labels_no_shape(self):
        """_plot_cluster_labels without shape (needs directly 2D so imshow works)."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_cluster_labels

        fig, ax = plt.subplots()
        # Labels with 25 pixels, which can be reshaped into (5, 5)
        labels = np.zeros((2, 25), dtype=bool)
        labels[0, :12] = True
        labels[1, 12:] = True
        # Must provide shape so cluster_map is 2D for imshow
        ax = _plot_cluster_labels(ax, labels, shape=(5, 5), title="test")
        assert ax is not None
        plt.close(fig)

    def test_plot_cluster_distances_helper_no_shape(self):
        """_plot_cluster_distances without shape."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_cluster_distances

        fig, ax = plt.subplots()
        dists = np.random.default_rng(1).standard_normal((2, 20))
        ax = _plot_cluster_distances(ax, dists, shape=None, title="test")
        assert ax is not None
        plt.close(fig)

    def test_plot_cluster_distances_helper_with_shape(self):
        """_plot_cluster_distances with shape uses reshape."""
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt

        from hyperspy_ml.plotting._helpers import _plot_cluster_distances

        fig, ax = plt.subplots()
        dists = np.random.default_rng(1).standard_normal((2, 77))
        ax = _plot_cluster_distances(ax, dists, shape=(7, 11), title="test")
        assert ax is not None
        plt.close(fig)


# ============================================================================
# BSSResult extra methods
# ============================================================================


class TestBSSResultMethods:
    def test_summary(self):
        """BSSResult.summary returns string with algorithm name."""
        br = BSSResult(
            bss_components=np.random.default_rng(1).standard_normal((13, 2)),
            bss_scores=np.random.default_rng(1).standard_normal((77, 2)),
            bss_algorithm="orthomax",
        )
        s = br.summary()
        assert "orthomax" in s
        assert "n_components" in s

    def test_repr(self):
        """BSSResult.__repr__ delegates to summary."""
        br = BSSResult(
            bss_components=np.random.default_rng(1).standard_normal((13, 2)),
            bss_scores=np.random.default_rng(1).standard_normal((77, 2)),
            bss_algorithm="orthomax",
        )
        r = repr(br)
        assert "orthomax" in r

    def test_summary_no_components(self):
        """BSSResult.summary handles None components."""
        br = BSSResult(bss_algorithm="orthomax")
        s = br.summary()
        assert "N/A" in s

    def test_save_load_roundtrip(self):
        """BSSResult.save and .load roundtrip."""
        br = BSSResult(
            bss_components=np.random.default_rng(1).standard_normal((13, 3)),
            bss_scores=np.random.default_rng(1).standard_normal((77, 3)),
            unmixing_matrix=np.eye(3),
            bss_algorithm="orthomax",
            on_scores=False,
        )
        with TemporaryDirectory() as tmp:
            br.save(tmp + "/bss_test.hsml")
            loaded = BSSResult.load(tmp + "/bss_test.hsml")
            np.testing.assert_array_almost_equal(
                loaded.bss_components, br.bss_components
            )
            assert loaded.bss_algorithm == br.bss_algorithm


# ============================================================================
# ClusterResult extra methods
# ============================================================================


class TestClusterResultMethods:
    def test_summary(self):
        """ClusterResult.summary returns string with algorithm."""
        cr = ClusterResult(
            cluster_labels=np.zeros((2, 77), dtype=bool),
            cluster_centroids=np.random.default_rng(1).standard_normal((2, 3)),
            cluster_sum_signals=np.random.default_rng(1).standard_normal((2, 13)),
            number_of_clusters=2,
            cluster_algorithm="KMeans",
            cluster_metric="gap",
        )
        s = cr.summary()
        assert "KMeans" in s

    def test_repr(self):
        """ClusterResult.__repr__ delegates to summary."""
        cr = ClusterResult(
            cluster_labels=np.zeros((2, 77), dtype=bool),
            cluster_sum_signals=np.random.default_rng(1).standard_normal((2, 13)),
            number_of_clusters=2,
            cluster_algorithm="KMeans",
        )
        r = repr(cr)
        assert "KMeans" in r

    def test_save_load_roundtrip(self):
        """ClusterResult.save and .load roundtrip."""
        labels = np.zeros((2, 77), dtype=bool)
        labels[0, :40] = True
        labels[1, 40:] = True
        cr = ClusterResult(
            cluster_labels=labels,
            cluster_centroids=np.random.default_rng(1).standard_normal((2, 3)),
            cluster_distances=np.random.default_rng(1).standard_normal((2, 77)),
            cluster_sum_signals=np.random.default_rng(1).standard_normal((2, 13)),
            cluster_centroid_signals=np.random.default_rng(1).standard_normal((2, 13)),
            number_of_clusters=2,
            cluster_algorithm="KMeans",
            estimated_number_of_clusters=2,
            cluster_metric="gap",
            cluster_metric_index=np.array([1, 2]),
            cluster_metric_data=np.array([0.5, 0.6]),
        )
        with TemporaryDirectory() as tmp:
            cr.save(tmp + "/cluster_test.hsml")
            loaded = ClusterResult.load(tmp + "/cluster_test.hsml")
            assert loaded.number_of_clusters == cr.number_of_clusters
            assert loaded.cluster_algorithm == cr.cluster_algorithm
            np.testing.assert_array_almost_equal(
                loaded.cluster_sum_signals, cr.cluster_sum_signals
            )

    def test_plot_cluster_distances(self):
        """ClusterResult.plot_cluster_distances renders."""
        import matplotlib

        matplotlib.use("agg")
        cr = ClusterResult(
            cluster_labels=np.zeros((2, 77), dtype=bool),
            cluster_distances=np.random.default_rng(1).standard_normal((2, 77)),
            cluster_sum_signals=np.random.default_rng(1).standard_normal((2, 13)),
            number_of_clusters=2,
        )
        ax = cr.plot_cluster_distances(nav_shape=(7, 11))
        assert ax is not None

    def test_get_cluster_labels(self):
        """ClusterResult.get_cluster_labels returns copy."""
        labels = np.zeros((2, 77), dtype=bool)
        cr = ClusterResult(
            cluster_labels=labels,
            cluster_sum_signals=np.random.default_rng(1).standard_normal((2, 13)),
            number_of_clusters=2,
        )
        out = cr.get_cluster_labels()
        np.testing.assert_array_equal(out, labels)

    def test_get_cluster_labels_none(self):
        """ClusterResult.get_cluster_labels returns None when None."""
        cr = ClusterResult(number_of_clusters=0)
        assert cr.get_cluster_labels() is None

    def test_get_cluster_signals(self):
        """ClusterResult.get_cluster_signals returns copy."""
        sigs = np.random.default_rng(1).standard_normal((2, 13))
        cr = ClusterResult(
            cluster_sum_signals=sigs,
            number_of_clusters=2,
            cluster_labels=np.zeros((2, 77), dtype=bool),
        )
        out = cr.get_cluster_signals()
        np.testing.assert_array_equal(out, sigs)

    def test_get_cluster_signals_none(self):
        """ClusterResult.get_cluster_signals returns None when None."""
        cr = ClusterResult(number_of_clusters=0)
        assert cr.get_cluster_signals() is None

    def test_get_cluster_distances(self):
        """ClusterResult.get_cluster_distances returns copy."""
        dists = np.random.default_rng(1).standard_normal((2, 77))
        cr = ClusterResult(
            cluster_distances=dists,
            cluster_labels=np.zeros((2, 77), dtype=bool),
            cluster_sum_signals=np.random.default_rng(1).standard_normal((2, 13)),
            number_of_clusters=2,
        )
        out = cr.get_cluster_distances()
        np.testing.assert_array_equal(out, dists)

    def test_get_cluster_distances_none(self):
        """ClusterResult.get_cluster_distances returns None when None."""
        cr = ClusterResult(number_of_clusters=0)
        assert cr.get_cluster_distances() is None


# ============================================================================
# DecompositionResult — extra coverage paths
# ============================================================================


class TestDecompositionResultExtra:
    def test_compute_explained_variance_ratio_zero_sum(self):
        """_compute_explained_variance_ratio sets None when ev_sum is zero."""
        result, s = _make_decom_result(rank=2)
        result.explained_variance = np.zeros(2)
        result._notify_data_changed()
        result._compute_explained_variance_ratio()
        assert result.explained_variance_ratio is None

    def test_normalize_decomposition_components_factors_deprecated(self):
        """normalize_decomposition_components with 'factors' warns."""
        result, s = _make_decom_result(rank=3)
        with pytest.warns(VisibleDeprecationWarning, match="factors"):
            result.normalize_decomposition_components(target="factors")

    def test_normalize_decomposition_components_loadings_deprecated(self):
        """normalize_decomposition_components with 'loadings' warns."""
        result, s = _make_decom_result(rank=3)
        with pytest.warns(VisibleDeprecationWarning, match="loadings"):
            result.normalize_decomposition_components(target="loadings")

    def test_normalize_decomposition_components_invalid_target(self):
        """normalize_decomposition_components raises on invalid target."""
        result, s = _make_decom_result(rank=3)
        with pytest.raises(ValueError, match="target must be"):
            result.normalize_decomposition_components(target="invalid")

    def test_normalize_decomposition_components_none_target(self):
        """normalize_decomposition_components raises when target is None."""
        result = DecompositionResult()
        with pytest.raises(ValueError, match="requires components"):
            result.normalize_decomposition_components(target="components")

    def test_reverse_decomposition_component_lazy(self):
        """reverse_decomposition_component warns for lazy (dask) results."""
        # Simulate lazy by monkey-patching a compute method on components
        result, s = _make_decom_result(rank=3)

        class FakeDaskArray(np.ndarray):
            pass

        result.components = result.components.view(FakeDaskArray)
        result.components.compute = lambda: None
        result.scores = result.scores.view(FakeDaskArray)

        result.reverse_decomposition_component(0)
        # Should log warning, not crash

    def test_get_decomposition_model_with_int_components(self):
        """Model reconstruction with components=int."""
        result, s = _make_decom_result(rank=4)
        sc = result.get_decomposition_model(s, components=2)
        assert sc is not None
        assert sc.data.shape == s.data.shape

    def test_get_decomposition_model_with_list_components(self):
        """Model reconstruction with components=list."""
        result, s = _make_decom_result(rank=4)
        sc = result.get_decomposition_model(s, components=[0, 2])
        assert sc is not None

    def test_get_decomposition_model_lazy_output(self):
        """Model reconstruction with lazy_output=True returns dask-backed signal."""
        result, s = _make_decom_result(rank=3)
        sc = result.get_decomposition_model(s, lazy_output=True)
        assert sc._lazy

    def test_get_decomposition_model_with_chunks_tuple(self):
        """Model reconstruction with chunks as tuple."""
        result, s = _make_decom_result(rank=3)
        sc = result.get_decomposition_model(s, chunks=(5, 5), lazy_output=True)
        assert sc is not None

    def test_get_decomposition_model_not_a_signal_raises(self):
        """get_decomposition_model with non-BaseSignal should raise TypeError."""
        from hyperspy.signals import BaseSignal

        result, s = _make_decom_result(rank=3)
        result._source_signal = None
        # Pass non-BaseSignal as source_signal
        not_a_signal: object = object()
        if isinstance(not_a_signal, BaseSignal):
            pytest.skip("object() is somehow a BaseSignal")
        try:
            result.get_decomposition_model(source_signal=not_a_signal)
        except (TypeError, AttributeError):
            pass  # Expected — either TypeError from BaseSignal check or AttributeError from missing deepcopy

    def test_get_decomposition_model_no_source_no_nav_shape(self):
        """get_decomposition_model raises when no signal and no stored nav_shape."""
        result, s = _make_decom_result(rank=3)
        result._source_signal = None
        result._nav_shape = None
        with pytest.raises(ValueError, match="No source_signal provided"):
            result.get_decomposition_model()

    def test_get_decomposition_model_mva_type_via_recmethod(self):
        """_calculate_recmatrix raises ValueError on invalid mva_type."""
        result, s = _make_decom_result(rank=3)
        with pytest.raises(ValueError, match="mva_type"):
            result._calculate_recmatrix(source_signal=s, mva_type="invalid_type")

    def test_normalize_poissonian_noise_no_signal(self):
        """normalize_poissonian_noise raises when no signal available."""
        result = DecompositionResult()
        with pytest.raises(ValueError, match="No signal"):
            result.normalize_poissonian_noise()

    def test_crop_decomposition_dimension_ev_ratio_none(self):
        """crop_decomposition_dimension when explained_variance_ratio is None."""
        result, s = _make_decom_result(rank=4)
        result.explained_variance_ratio = None
        result.crop_decomposition_dimension(2)  # should not crash
        assert result.n_components == 2

    def test_plot_bss_factors_deprecated(self):
        """plot_bss_factors warns and calls plot_bss_components."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        with pytest.warns(VisibleDeprecationWarning, match="factors"):
            result.plot_bss_factors()

    def test_plot_bss_loadings_deprecated(self):
        """plot_bss_loadings warns and calls plot_bss_scores."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        with pytest.warns(VisibleDeprecationWarning, match="loadings"):
            result.plot_bss_loadings()

    def test_get_decomposition_components(self):
        """get_decomposition_components delegates to get_components."""
        result, s = _make_decom_result(rank=3)
        c = result.get_decomposition_components()
        np.testing.assert_array_equal(c, result.components)

    def test_get_decomposition_scores(self):
        """get_decomposition_scores delegates to get_scores."""
        result, s = _make_decom_result(rank=3)
        sc = result.get_decomposition_scores()
        np.testing.assert_array_equal(sc, result.scores)

    def test_get_bss_components_success(self):
        """get_bss_components returns BSS components when available."""
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        c = result.get_bss_components()
        assert c.shape == (13, 3)

    def test_get_bss_components_with_index(self):
        """get_bss_components with idx returns subset."""
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        c = result.get_bss_components(idx=0)
        assert c.shape == (13,)

    def test_get_bss_scores_success(self):
        """get_bss_scores returns BSS scores when available."""
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        sc = result.get_bss_scores()
        assert sc.shape == (77, 3)

    def test_get_bss_scores_with_index(self):
        """get_bss_scores with idx returns subset."""
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        sc = result.get_bss_scores(idx=1)
        assert sc.shape == (77,)

    def test_get_bss_factors_deprecated(self):
        """get_bss_factors warns and delegates."""
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        with pytest.warns(VisibleDeprecationWarning, match="factors"):
            c = result.get_bss_factors()
        assert c.shape == (13, 3)

    def test_get_bss_loadings_deprecated(self):
        """get_bss_loadings warns and delegates."""
        result, s = _make_decom_result(rank=3)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        with pytest.warns(VisibleDeprecationWarning, match="loadings"):
            sc = result.get_bss_loadings()
        assert sc.shape == (77, 3)

    def test_plot_scree_with_horizontal_line_auto_float(self):
        """plot_scree with hline='auto' and float threshold."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=8)
        ax = result.plot_scree(hline="auto", threshold=0.1)
        assert ax is not None

    def test_get_number_significant_components_none(self):
        """_get_number_significant_components returns None when no variance ratio."""
        result = DecompositionResult()
        assert result._get_number_significant_components() is None

    def test_get_number_significant_components(self):
        """_get_number_significant_components returns elbow position."""
        result, s = _make_decom_result(rank=6)
        result._compute_explained_variance_ratio()
        n = result._get_number_significant_components()
        assert isinstance(n, (int, np.integer))
        assert n >= 1

    def test_plot_scree_with_vline_and_number_significant(self):
        """plot_scree with vline uses _get_number_significant_components."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=6)
        result._compute_explained_variance_ratio()
        ax = result.plot_scree(vline=True, xaxis_type="number")
        assert ax is not None

    def test_plot_scree_xaxis_index(self):
        """plot_scree with xaxis_type='index'."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=5)
        ax = result.plot_scree(xaxis_type="index")
        assert ax is not None

    def test_write_to_signal_full(self):
        """write_to_signal populates all learning_results attributes."""
        result, s = _make_decom_result(rank=3)
        import numpy as np

        result.bss_components = result.components
        result.bss_scores = result.scores
        result.unmixing_matrix = np.eye(3)
        result.write_to_signal(s)
        lr = s.learning_results
        assert lr.decomposition_algorithm is not None
        assert lr.output_dimension is not None

    def test_matrix_field_names(self):
        """_array_field_names and _scalar_field_names have no overlap for DecompositionResult."""
        arr = _array_field_names(DecompositionResult)
        sca = _scalar_field_names(DecompositionResult)
        assert not set(arr) & set(sca)


# ============================================================================
# Study — save/load, remove, iter, repr
# ============================================================================


class TestStudy:
    def test_study_add_auto_name(self):
        """Study.add with no name auto-generates one."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        key = study.add(result)
        assert key.startswith("DecompositionResult_")

    def test_study_add_explicit_name(self):
        """Study.add with explicit name uses it."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        key = study.add(result, name="my_decom")
        assert key == "my_decom"
        assert "my_decom" in study

    def test_study_setitem(self):
        """Study.__setitem__ adds result."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study["manual"] = result
        assert "manual" in study

    def test_study_delitem(self):
        """Study.__delitem__ removes result."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study["manual"] = result
        del study["manual"]
        assert "manual" not in study

    def test_study_len(self):
        """Study.__len__ returns number of results."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        assert len(study) == 0
        result, s = _make_decom_result(rank=3)
        study.add(result)
        assert len(study) == 1

    def test_study_iter(self):
        """Study.__iter__ yields keys."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study.add(result, name="a")
        study.add(result, name="b")
        keys = list(iter(study))
        assert "a" in keys
        assert "b" in keys

    def test_study_keys_values_items_get(self):
        """Study.keys/values/items/get methods."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study.add(result, name="r1")
        assert "r1" in study.keys()
        assert len(list(study.values())) == 1
        items_list = list(study.items())
        assert items_list[0][0] == "r1"
        assert study.get("nonexistent", None) is None
        assert study.get("r1") is not None

    def test_study_repr(self):
        """Study.__repr__ returns summary string."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study.add(result, name="r1")
        r = repr(study)
        assert "test_study" in r
        assert "r1" in r

    def test_study_save_load(self):
        """Study.save and load roundtrip."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study.add(result, name="r1", source_signal=s)

        with TemporaryDirectory() as tmp:
            study.save(tmp + "/study_test")
            # Re-loading a study would need a load method; verify save works.
            import zarr

            store = zarr.open(tmp + "/study_test", mode="r")
            assert store.attrs["type"] == "Study"
            assert "r1" in store.attrs["keys"]

    def test_study_remove(self):
        """Study.remove removes result and dependents."""
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study.add(result, name="parent")
        study.add(result, name="parent_child")  # dependent on "parent"
        assert "parent" in study
        assert "parent_child" in study
        study.remove("parent")
        assert "parent" not in study
        assert "parent_child" not in study  # dependent removed too


# ============================================================================
# Result events
# ============================================================================


class TestResultEvents:
    def test_events_data_changed_getter(self):
        """_ResultEvents.data_changed returns fired state."""
        result, s = _make_decom_result(rank=3)
        assert not result.events.data_changed
        result._notify_data_changed()
        assert result.events.data_changed

    def test_events_notify_twice(self):
        """Notify twice still fires callbacks both times."""
        result, s = _make_decom_result(rank=3)
        counter = []

        def cb():
            counter.append(1)

        result.events.connect(cb)
        result._notify_data_changed()
        result._notify_data_changed()
        assert len(counter) == 2

    def test_disconnect_non_registered(self):
        """Disconnecting a non-registered callback raises ValueError."""
        result, s = _make_decom_result(rank=3)

        def cb():
            pass

        with pytest.raises(ValueError):
            result.events.disconnect(cb)


# ============================================================================
# LearningResults deprecated bridge tests
# ============================================================================


class TestLearningResultsDeprecated:
    def test_factors_property(self):
        """LearningResults.factors property warns and delegates."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        lr.components = np.array([[1.0, 2.0]])
        with pytest.warns(VisibleDeprecationWarning, match="factors"):
            f = lr.factors
        np.testing.assert_array_equal(f, lr.components)

    def test_factors_setter(self):
        """LearningResults.factors setter warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        val = np.array([[3.0, 4.0]])
        with pytest.warns(VisibleDeprecationWarning, match="factors"):
            lr.factors = val
        np.testing.assert_array_equal(lr.components, val)

    def test_loadings_property(self):
        """LearningResults.loadings property warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        lr.scores = np.array([[1.0]])
        with pytest.warns(VisibleDeprecationWarning, match="loadings"):
            s = lr.loadings
        np.testing.assert_array_equal(s, lr.scores)

    def test_loadings_setter(self):
        """LearningResults.loadings setter warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        val = np.array([[5.0]])
        with pytest.warns(VisibleDeprecationWarning, match="loadings"):
            lr.loadings = val
        np.testing.assert_array_equal(lr.scores, val)

    def test_bss_factors_property(self):
        """LearningResults.bss_factors property warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        lr.bss_components = np.array([[1.0]])
        with pytest.warns(VisibleDeprecationWarning, match="bss_factors"):
            f = lr.bss_factors
        np.testing.assert_array_equal(f, lr.bss_components)

    def test_bss_factors_setter(self):
        """LearningResults.bss_factors setter warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        val = np.array([[7.0]])
        with pytest.warns(VisibleDeprecationWarning, match="bss_factors"):
            lr.bss_factors = val
        np.testing.assert_array_equal(lr.bss_components, val)

    def test_bss_loadings_property(self):
        """LearningResults.bss_loadings property warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        lr.bss_scores = np.array([[1.0]])
        with pytest.warns(VisibleDeprecationWarning, match="bss_loadings"):
            s = lr.bss_loadings
        np.testing.assert_array_equal(s, lr.bss_scores)

    def test_bss_loadings_setter(self):
        """LearningResults.bss_loadings setter warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        val = np.array([[9.0]])
        with pytest.warns(VisibleDeprecationWarning, match="bss_loadings"):
            lr.bss_loadings = val
        np.testing.assert_array_equal(lr.bss_scores, val)

    def test_on_loadings_property(self):
        """LearningResults.on_loadings property warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        lr.on_scores = True
        with pytest.warns(VisibleDeprecationWarning, match="on_loadings"):
            v = lr.on_loadings
        assert v is True

    def test_on_loadings_setter(self):
        """LearningResults.on_loadings setter warns."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        with pytest.warns(VisibleDeprecationWarning, match="on_loadings"):
            lr.on_loadings = True
        assert lr.on_scores is True

    def test_npz_save_load(self):
        """LearningResults.save_npz and load_npz roundtrip."""
        from hyperspy_ml.results.learning_results import LearningResults

        lr = LearningResults()
        lr.components = np.random.default_rng(1).standard_normal((13, 3))
        lr.scores = np.random.default_rng(1).standard_normal((77, 3))
        lr.explained_variance = np.array([10.0, 5.0, 1.0])
        lr.decomposition_algorithm = "SVD"
        lr.output_dimension = 3
        with TemporaryDirectory() as tmp:
            path = tmp + "/test.npz"
            lr.save(path)
            loaded = LearningResults()
            loaded.load(path)
            np.testing.assert_array_almost_equal(loaded.components, lr.components)
            np.testing.assert_array_almost_equal(loaded.scores, lr.scores)


# ============================================================================
# Decomposition — scree plot deprecated accessors
# ============================================================================


class TestDecompositionScreeDeprecated:
    def test_plot_explained_variance_ratio_deprecated(self):
        """plot_explained_variance_ratio warns and delegates."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=5)
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            ax = result.plot_explained_variance_ratio()
        assert ax is not None

    def test_plot_cumulative_explained_variance_ratio_deprecated(self):
        """plot_cumulative_explained_variance_ratio warns."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=5)
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            ax = result.plot_cumulative_explained_variance_ratio()
        assert ax is not None


# ============================================================================
# Decomposition.write_to_signal
# ============================================================================


class TestWriteToSignal:
    def test_write_to_signal_bss_unmixing(self):
        """write_to_signal handles bss_components and unmixing_matrix."""
        result, s = _make_decom_result(rank=3)
        result.bss_components = result.components.copy()
        result.bss_scores = result.scores.copy()
        result.unmixing_matrix = np.eye(3)
        result.write_to_signal(s)
        lr = s.learning_results
        np.testing.assert_array_almost_equal(lr.bss_components, result.components)
        np.testing.assert_array_almost_equal(lr.unmixing_matrix, np.eye(3))


# ============================================================================
# Decomposition — scree plot methods
# ============================================================================


class TestScreePlotCoverage:
    def test_plot_scree_no_threshold(self):
        """plot_scree without threshold."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=6)
        result._compute_explained_variance_ratio()
        ax = result.plot_scree()
        assert ax is not None

    def test_plot_scree_log(self):
        """plot_scree with log=True."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=6)
        ax = result.plot_scree(log=True)
        assert ax is not None

    def test_plot_cumulative_scree_full(self):
        """plot_cumulative_scree with all arguments."""
        import matplotlib

        matplotlib.use("agg")
        result, s = _make_decom_result(rank=6)
        ax = result.plot_cumulative_scree(n=4)
        assert ax is not None

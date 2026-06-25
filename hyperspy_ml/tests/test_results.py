# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Tests for results data model with .hsml Zarr save/load (Task 7a).

ALL-DIFFERENT dimensions (7, 11, 13) used throughout.
"""

from __future__ import annotations

import sys
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import (
    BSSResult,
    ClusterResult,
    DecompositionResult,
    load_result,
    save_result,
)
from hyperspy_ml.stages.decomposition import Decomposition

zarr = pytest.importorskip("zarr", reason="zarr required for .hsml tests")


def _make_decom_result(rng_seed=42, rank=3):
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
# Events
# ============================================================================


class TestEvents:
    def test_data_changed_fires_callback(self):
        result, s = _make_decom_result()
        fired = []

        def cb():
            fired.append(1)

        result.events.connect(cb)
        result._notify_data_changed()
        assert len(fired) == 1

    def test_disconnect(self):
        result, s = _make_decom_result()
        fired = []

        def cb():
            fired.append(1)

        result.events.connect(cb)
        result.events.disconnect(cb)
        result._notify_data_changed()
        assert len(fired) == 0


# ============================================================================
# DecompositionResult — full attributes
# ============================================================================


class TestDecompositionResultAttrs:
    def test_all_learning_results_attrs_present(self):
        result = DecompositionResult()
        for attr in (
            "components",
            "scores",
            "bss_components",
            "bss_scores",
            "explained_variance",
            "explained_variance_ratio",
            "mean",
            "bH",
            "aG",
            "n_components",
            "centre",
            "algorithm",
            "params",
            "number_significant_components",
            "poissonian_noise_normalized",
            "output_dimension",
            "navigation_mask",
            "signal_mask",
            "unmixing_matrix",
            "bss_algorithm",
            "unfolded",
            "original_shape",
        ):
            assert hasattr(result, attr), f"missing {attr}"

    def test_write_to_signal_populates_learning_results(self):
        result, s = _make_decom_result()
        result.write_to_signal(s)
        lr = s.learning_results
        assert lr.components is not None
        assert lr.scores is not None
        assert lr.decomposition_algorithm == "SVD"
        assert lr.output_dimension == 3
        assert lr.centre is None

    def test_summary_contains_algorithm(self):
        result, s = _make_decom_result()
        text = result.summary()
        assert "algorithm=SVD" in text

    def test_bss_attrs_saved_to_learning_results(self):
        result, s = _make_decom_result()
        # Simulate BSS results
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        result.unmixing_matrix = R
        result.write_to_signal(s)
        lr = s.learning_results
        assert lr.bss_components is not None
        assert lr.bss_scores is not None


# ============================================================================
# .hsml Zarr round-trip
# ============================================================================


class TestHSMLRoundTrip:
    def test_decom_round_trip(self):
        result, s = _make_decom_result()
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/test.hsml"
            save_result(result, path, source_signal=s)
            loaded = load_result(path)

        assert isinstance(loaded, DecompositionResult)
        np.testing.assert_allclose(loaded.components, result.components)
        np.testing.assert_allclose(loaded.scores, result.scores)
        np.testing.assert_allclose(loaded.explained_variance, result.explained_variance)
        assert loaded.algorithm == "SVD"
        assert loaded.n_components == 3

    def test_bss_round_trip(self):
        result, s = _make_decom_result()
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = np.asarray(result.components @ R)
        result.bss_scores = np.asarray(result.scores @ R.T)

        bss = BSSResult(
            bss_components=result.bss_components,
            bss_scores=result.bss_scores,
            unmixing_matrix=R,
            bss_algorithm="orthomax",
            on_scores=False,
        )
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/bss.hsml"
            save_result(bss, path)
            loaded = load_result(path)

        assert isinstance(loaded, BSSResult)
        np.testing.assert_allclose(loaded.bss_components, bss.bss_components)
        assert loaded.bss_algorithm == "orthomax"

    def test_cluster_round_trip(self):
        labels = np.zeros((3, 77), dtype=bool)
        labels[0, :20] = True
        labels[1, 20:50] = True
        labels[2, 50:] = True
        cr = ClusterResult(
            cluster_labels=labels,
            cluster_centroids=np.random.default_rng(1).standard_normal((3, 5)),
            number_of_clusters=3,
            cluster_algorithm="kmeans",
        )
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/cluster.hsml"
            save_result(cr, path)
            loaded = load_result(path)

        assert isinstance(loaded, ClusterResult)
        np.testing.assert_array_equal(loaded.cluster_labels, labels)
        assert loaded.number_of_clusters == 3

    def test_attrs_preserved(self):
        result, s = _make_decom_result()
        result.algorithm = "sklearn_pca"
        result.centre = "navigation"
        result.poissonian_noise_normalized = True
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/attrs.hsml"
            save_result(result, path, source_signal=s)
            loaded = load_result(path)

        assert loaded.algorithm == "sklearn_pca"
        assert loaded.centre == "navigation"
        assert loaded.poissonian_noise_normalized is True

    def test_params_preserved(self):
        result, s = _make_decom_result()
        result.params = {"custom": 42, "flag": True}
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/params.hsml"
            save_result(result, path)
            loaded = load_result(path)

        assert loaded.params == {"custom": 42, "flag": True}


class TestHSMLVersionCheck:
    def test_major_version_mismatch_raises(self):
        result, s = _make_decom_result()
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/version.hsml"
            save_result(result, path)
            import zarr

            store = zarr.open(path, mode="r+")
            store.attrs["hsml_version"] = "2.0"

            with pytest.raises(OSError, match="version 2.0"):
                load_result(path)


class TestHSMLSourceDeduplication:
    def test_same_source_shares_hash(self):
        result1, s1 = _make_decom_result(42)
        result2, s2 = _make_decom_result(42)  # same seed → same data
        with TemporaryDirectory() as tmp:
            p1, p2 = f"{tmp}/r1.hsml", f"{tmp}/r2.hsml"
            save_result(result1, p1, source_signal=s1)
            save_result(result2, p2, source_signal=s2)
            from zarr import open as zopen

            s1 = zopen(p1)
            s2_obj = zopen(p2)
            assert s1.attrs["source_hash"] == s2_obj.attrs["source_hash"]


class TestHSMLConvenienceMethods:
    def test_save_load_methods(self):
        result, s = _make_decom_result()
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/conv.hsml"
            result.save(path)
            loaded = DecompositionResult.load(path)
        np.testing.assert_allclose(loaded.components, result.components)


# ============================================================================
# Events on BSS and Cluster results
# ============================================================================


class TestBSSClusterEvents:
    def test_bss_has_events(self):
        r = BSSResult()
        assert hasattr(r, "events")

    def test_cluster_has_events(self):
        r = ClusterResult()
        assert hasattr(r, "events")

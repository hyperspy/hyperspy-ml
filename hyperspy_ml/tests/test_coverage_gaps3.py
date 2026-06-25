# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Additional quick-win coverage tests to push past 85%."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.stages.bss import BSS
from hyperspy_ml.stages.decomposition import Decomposition

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")
dask = importlib.util.find_spec("dask")
skip_dask = pytest.mark.skipif(dask is None, reason="dask not installed")


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


class TestBSSMore:
    @skip_sklearn
    def test_orthomax_diff_axes(self):
        result, s = _make_decom_result(rank=4)
        stage = BSS(
            number_of_components=3,
            algorithm="orthomax",
            diff_axes=0,
            diff_order=1,
            print_info=False,
        )
        bss_result, _extra = stage.fit_transform(result)
        assert bss_result.bss_components is not None

    @skip_sklearn
    def test_orthomax_reverse_component_criterion_scores(self):
        result, s = _make_decom_result(rank=4)
        stage = BSS(
            number_of_components=3,
            algorithm="orthomax",
            reverse_component_criterion="scores",
            print_info=False,
        )
        bss_result, _extra = stage.fit_transform(result)
        assert bss_result.bss_components is not None

    @skip_sklearn
    def test_bss_raises_without_decomposition(self):
        from hyperspy_ml.results.base import DecompositionResult

        empty = DecompositionResult()
        stage = BSS(number_of_components=2, algorithm="orthomax", print_info=False)
        with pytest.raises(AttributeError, match="decomposition"):
            stage.fit_transform(empty)


class TestDecompositionMore:
    def test_decomposition_unknown_algorithm_raises(self):
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="nonexistent_algo", output_dimension=3, print_info=False
        )
        with pytest.raises(ValueError, match="algorithm"):
            stage.fit_transform(s)

    def test_centre_invalid_raises(self):
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, centre="invalid", print_info=False
        )
        with pytest.raises(ValueError, match="centre"):
            stage.fit_transform(s)

    @skip_sklearn
    def test_decomposition_estimator_without_components_raises(self):
        from sklearn.cluster import KMeans

        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        estim = KMeans(n_clusters=3, random_state=42, n_init=10)
        stage = Decomposition(algorithm=estim, output_dimension=None, print_info=False)
        with pytest.raises(AttributeError, match="components_"):
            stage.fit_transform(s)

    @skip_sklearn
    def test_signal_dimension_guard(self):
        s = Signal1D(np.random.default_rng(1).standard_normal((1, 13)))
        stage = Decomposition(algorithm="SVD", output_dimension=2, print_info=False)
        with pytest.raises(ValueError, match="navigation"):
            stage.fit_transform(s)

    @skip_dask
    def test_lazy_num_chunks(self):
        import dask.array as da

        data = da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        s = Signal1D(data.reshape(7, 11, 13)).as_lazy()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            svd_solver="randomized",
            num_chunks=4,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_decomposition_output_dimension_gt_rank(self):
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(algorithm="SVD", output_dimension=13, print_info=False)
        result = stage.fit_transform(s)
        assert result.n_components == 13

    def test_normalise_navigation_mask_transposed(self):
        nm = np.ones((11, 7), dtype=bool)
        stage = Decomposition(algorithm="SVD", output_dimension=3)
        result = stage._normalise_navigation_mask(nm)
        assert result is not None


class TestDecompositionModelMore:
    def test_get_bss_model_basic(self):
        result, s = _make_decom_result(rank=3)
        from numpy.linalg import qr

        R = qr(np.random.default_rng(1).standard_normal((3, 3)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T
        sc = result.get_bss_model(s)
        assert sc is not None

    def test_get_bss_model_raises_without_bss(self):
        result, s = _make_decom_result(rank=3)
        with pytest.raises(ValueError, match="No bss"):
            result.get_bss_model(s)

    def test_get_decomposition_model_lazy_output_false(self):
        result, s = _make_decom_result(rank=3)
        sc = result.get_decomposition_model(s, lazy_output=False)
        assert not sc._lazy
        assert sc.data.shape == s.data.shape

    def test_get_decomposition_model_chunks_int(self):
        result, s = _make_decom_result(rank=3)
        sc = result.get_decomposition_model(s, chunks=5, lazy_output=True)
        assert sc is not None


class TestResultsUtilsMore:
    def test_apply_preprocessing_no_ops(self):
        from hyperspy_ml.utils.preprocessing import apply_preprocessing

        data = np.random.default_rng(1).standard_normal((20, 10))
        dc, mean, ag, bh = apply_preprocessing(data, centre=None)
        assert dc is data
        assert mean is None

    def test_apply_preprocessing_kk_with_numpy(self):
        from hyperspy_ml.utils.preprocessing import apply_preprocessing

        data = np.abs(np.random.default_rng(1).standard_normal((20, 10))) + 0.1
        dc, mean, ag, bh = apply_preprocessing(
            data, normalize_poissonian_noise=True, ndim=1, sdim=1
        )
        assert dc.shape == data.shape

    def test_nan_expand_rows_basic(self):
        from hyperspy_ml.utils.preprocessing import _nan_expand_rows

        arr = np.ones((15, 5))
        mask = np.zeros(20, dtype=bool)
        mask[5:10] = True
        expanded = _nan_expand_rows(arr, mask, 20)
        assert expanded.shape == (20, 5)
        assert np.isnan(expanded[5:10]).all()

    def test_reproject_navigation(self):
        from hyperspy_ml.utils.preprocessing import _reproject_navigation_scores

        D = np.random.default_rng(1).standard_normal((20, 10))
        components = np.linalg.svd(D, full_matrices=False)[2].T[:, :4]
        scores = _reproject_navigation_scores(D, components)
        assert scores.shape == (20, 4)

    def test_reproject_signal(self):
        from hyperspy_ml.utils.preprocessing import _reproject_signal_components

        D = np.random.default_rng(1).standard_normal((20, 10))
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        scores = U[:, :4]
        components = _reproject_signal_components(D, scores)
        assert components.shape == (10, 4)

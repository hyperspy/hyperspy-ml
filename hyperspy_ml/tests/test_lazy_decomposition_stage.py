# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Tests for Decomposition().fit_transform() on lazy (dask) signals.

Extracted from test_lazy_decomposition.py (Task 4b).
ALL-DIFFERENT dimensions (7, 11, 13) used throughout.
"""

import sys as _sys

import dask.array as da
import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in _sys.path:
    _sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.stages.decomposition import Decomposition  # noqa: E402

skip_sklearn = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)


def _make_lazy_signal(seed=42, nav_y=7, nav_x=11, sig=13, rank=3):
    """Create a lazy Signal1D with ALL-DIFFERENT dimensions and known rank."""
    rng = np.random.default_rng(seed)
    nav_size = nav_y * nav_x
    U = rng.standard_normal((nav_size, rank))
    V = rng.standard_normal((sig, rank))
    X = U @ V.T
    return Signal1D(X.reshape(nav_y, nav_x, sig)).as_lazy(), nav_y, nav_x, sig


class TestDecompositionStageLazy:
    """fit_transform() on lazy signals."""

    def test_basic_svd_randomized(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.components.shape == (sig, 3)
        assert result.scores.shape == (n_y * n_x, 3)
        assert result.explained_variance is not None

    def test_svd_full_returns_dask_arrays(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        stage = Decomposition(
            algorithm="SVD", svd_solver="full", output_dimension=3, print_info=False
        )
        result = stage.fit_transform(s)
        assert isinstance(result.components, da.Array)

    def test_keenan_kotula_scaling(self):
        rng = np.random.default_rng(7)
        data = rng.integers(1, 100, size=(15, 20)).astype(float)
        s = Signal1D(data).as_lazy()
        stage = Decomposition(
            normalize_poissonian_noise=True,
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=2,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.bH is not None
        assert result.aG is not None

    def test_signal_mask(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        sm = np.zeros(sig, dtype=bool)
        sm[-3:] = True
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            signal_mask=sm,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert np.all(np.isnan(result.components[-3:]))

    def test_navigation_mask(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        nm = np.zeros(n_y * n_x, dtype=bool)
        nm[:5] = True
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            navigation_mask=nm,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert np.all(np.isnan(result.scores[:5]))

    def test_lazy_vs_eager_agreement(self):
        rng = np.random.default_rng(99)
        data = rng.standard_normal((20, 3)) @ rng.standard_normal((3, 30))
        s_nl = Signal1D(data)
        s_lz = Signal1D(data).as_lazy()
        stage = Decomposition(algorithm="SVD", output_dimension=3, print_info=False)
        r_nl = stage.fit_transform(s_nl)
        r_lz = stage.fit_transform(s_lz)
        np.testing.assert_allclose(
            r_nl.explained_variance, r_lz.explained_variance, rtol=1e-6
        )

    def test_reproject_navigation(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        nm = np.zeros(n_y * n_x, dtype=bool)
        nm[:5] = True
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=3,
            navigation_mask=nm,
            reproject="navigation",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert np.all(np.isfinite(result.scores))

    @skip_sklearn
    def test_incremental_basic(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.components.shape == (sig, 3)


class TestDecompositionStageLazyIncremental:
    """Incremental SVD on lazy signals."""

    @skip_sklearn
    def test_basic(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    @skip_sklearn
    def test_reconstruction(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            print_info=False,
        )
        result = stage.fit_transform(s)
        recon = result.scores @ result.components.T
        orig = s.data.compute().reshape(n_y * n_x, sig)
        assert np.linalg.norm(recon - orig) < 1e-10

    @skip_sklearn
    def test_with_centre(self):
        s, n_y, n_x, sig = _make_lazy_signal()
        stage = Decomposition(
            algorithm="SVD",
            svd_solver="incremental",
            output_dimension=3,
            centre="navigation",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.mean is not None

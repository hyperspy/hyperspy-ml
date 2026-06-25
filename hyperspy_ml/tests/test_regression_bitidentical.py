# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy ML.

"""Bit-identical regression tests for SVD, K-K scaling, and reconstruction.

ALL-DIFFERENT dimensions (7, 11, 13) used throughout so that axis
reversals are immediately visible.
"""

from __future__ import annotations

import sys

import dask.array as da
import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.stages.decomposition import Decomposition  # noqa: E402
from hyperspy_ml.utils.preprocessing import _keenan_kotula_scale  # noqa: E402

RTOL = atol = 1e-10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _fixture_low_rank_data():
    """Deterministic low-rank data with all-different dims (7, 11, 13)."""
    rng = np.random.default_rng(42)
    rank = 3
    nav_y, nav_x, sig = 7, 11, 13
    nav_size = nav_y * nav_x
    U = rng.standard_normal((nav_size, rank))
    V = rng.standard_normal((sig, rank))
    X = U @ V.T
    return X, nav_y, nav_x, sig, rank


def _make_signal(rng_seed=42, rank=3):
    """Signal1D with all-different dims (7, 11, 13)."""
    rng = np.random.default_rng(rng_seed)
    nav_y, nav_x, sig = 7, 11, 13
    nav_size = nav_y * nav_x
    U = rng.standard_normal((nav_size, rank))
    V = rng.standard_normal((sig, rank))
    X = U @ V.T
    s = Signal1D(X.reshape(nav_y, nav_x, sig))
    return s, nav_y, nav_x, sig, rank


def _make_result(s=None, rng_seed=42, rank=3):
    """Return a DecompositionResult from SVD on *s* (creates if None)."""
    if s is None:
        s, n_y, n_x, sig, rank = _make_signal(rng_seed, rank)
    else:
        n_y, n_x, sig = 7, 11, 13
    stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
    result = stage.fit_transform(s)
    result._nav_shape = s.axes_manager.navigation_shape[::-1]
    result._source_signal = s
    return result, s, n_y, n_x, sig, rank


# ---------------------------------------------------------------------------
# SVD — deterministic bit-identical
# ---------------------------------------------------------------------------


class TestSVDBitIdentical:
    """SVD results must be deterministic and bit-identical across runs."""

    def test_svd_deterministic_repeated_runs(self, _fixture_low_rank_data):
        """Same SVD on same data produces bit-identical results across runs."""
        X, n_y, n_x, sig, rank = _fixture_low_rank_data
        s = Signal1D(X.reshape(n_y, n_x, sig))

        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        r1 = stage.fit_transform(s)
        r2 = stage.fit_transform(s)

        np.testing.assert_allclose(r1.components, r2.components, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(r1.scores, r2.scores, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(
            r1.explained_variance, r2.explained_variance, rtol=RTOL, atol=atol
        )

    def test_svd_deterministic_same_seed(self):
        """Same seed produces identical SVD results."""
        s1, n_y, n_x, sig, rank = _make_signal(rng_seed=42, rank=3)
        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        r1 = stage.fit_transform(s1)

        # Reset and recreate with same seed
        s2, _, _, _, _ = _make_signal(rng_seed=42, rank=3)
        r2 = stage.fit_transform(s2)

        np.testing.assert_allclose(r1.components, r2.components, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(r1.scores, r2.scores, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(
            r1.explained_variance, r2.explained_variance, rtol=RTOL, atol=atol
        )


# ---------------------------------------------------------------------------
# K-K scaling — numpy vs dask bit-identical
# ---------------------------------------------------------------------------


class TestKeenanKotulaBitIdentical:
    """K-K Poisson-noise scaling: numpy and dask paths must match exactly."""

    def test_kk_numpy_dask_bit_identical(self):
        """K-K scaling via numpy and dask produces bit-identical results."""
        rng = np.random.default_rng(101)
        data = rng.poisson(50, (7, 11, 13)).astype(float)
        nm = rng.random((7, 11)) < 0.1
        sm = rng.random(13) < 0.1

        # Numpy path
        s_np, aG_np, bH_np = _keenan_kotula_scale(
            data, navigation_mask=nm, signal_mask=sm, ndim=2, sdim=1
        )

        # Dask path
        d_data = da.from_array(data, chunks=(3, 5, 7))
        s_da, aG_da, bH_da = _keenan_kotula_scale(
            d_data, navigation_mask=nm, signal_mask=sm, ndim=2, sdim=1
        )

        np.testing.assert_allclose(s_da.compute(), s_np, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(aG_da.compute(), aG_np, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(bH_da.compute(), bH_np, rtol=RTOL, atol=atol)

    def test_kk_no_masks_numpy_dask_bit_identical(self):
        """K-K scaling without masks: numpy and dask bit-identical."""
        rng = np.random.default_rng(202)
        data = rng.poisson(50, (7, 11, 13)).astype(float)

        s_np, aG_np, bH_np = _keenan_kotula_scale(
            data, navigation_mask=None, signal_mask=None, ndim=2, sdim=1
        )
        d_data = da.from_array(data, chunks=(3, 5, 7))
        s_da, aG_da, bH_da = _keenan_kotula_scale(
            d_data, navigation_mask=None, signal_mask=None, ndim=2, sdim=1
        )

        np.testing.assert_allclose(s_da.compute(), s_np, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(aG_da.compute(), aG_np, rtol=RTOL, atol=atol)
        np.testing.assert_allclose(bH_da.compute(), bH_np, rtol=RTOL, atol=atol)


# ---------------------------------------------------------------------------
# Model reconstruction — bit-identical
# ---------------------------------------------------------------------------


class TestModelReconstructionBitIdentical:
    """Model reconstruction must be deterministic and bit-identical."""

    def test_reconstruction_full_deterministic(self):
        """Full reconstruction is bit-identical across two SVD runs."""
        result1, s1, n_y, n_x, sig, rank = _make_result(rng_seed=42)
        result2, s2, _, _, _, _ = _make_result(rng_seed=42)

        model1 = result1.get_decomposition_model()
        model2 = result2.get_decomposition_model()

        np.testing.assert_allclose(model1.data, model2.data, rtol=RTOL, atol=atol)

    def test_reconstruction_matches_scores_components(self):
        """Model = scores @ components.T to bit-identical precision."""
        result, s, n_y, n_x, sig, rank = _make_result()
        model = result.get_decomposition_model()

        recon_manual = result.scores @ result.components.T
        np.testing.assert_allclose(
            model.data.reshape(-1, sig), recon_manual, rtol=RTOL, atol=atol
        )

    def test_reconstruction_signal_identity_rank(self):
        """Rank-full SVD reconstruction recovers the original signal."""
        s, n_y, n_x, sig, rank = _make_signal(rng_seed=123)
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=sig,  # full-rank
            svd_solver="full",
            print_info=False,
        )
        result = stage.fit_transform(s)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model()

        np.testing.assert_allclose(model.data, s.data, rtol=RTOL, atol=atol)

    def test_reconstruction_with_subset_deterministic(self):
        """Component-subset reconstruction is deterministic."""
        result1, s1, n_y, n_x, sig, rank = _make_result(rng_seed=42)
        result2, s2, _, _, _, _ = _make_result(rng_seed=42)

        model1 = result1.get_decomposition_model(components=2)
        model2 = result2.get_decomposition_model(components=2)

        np.testing.assert_allclose(model1.data, model2.data, rtol=RTOL, atol=atol)

    def test_reconstruction_close_to_original(self):
        """Rank-3 SVD of rank-3 data recovers input approximately (no mean)."""
        s, n_y, n_x, sig, rank = _make_signal(rng_seed=77)
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=rank,
            svd_solver="full",
            print_info=False,
        )
        result = stage.fit_transform(s)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model()

        recon = model.data.reshape(-1, sig)
        orig = s.data.reshape(-1, sig)

        # Rank-3 approximation of rank-3 data should be exact
        # (small tolerance due to SVD numerics)
        np.testing.assert_allclose(recon, orig, rtol=1e-8, atol=1e-8)

    def test_reconstruction_lazy_vs_eager_bit_identical(self):
        """Lazy reconstruction must yield same data as eager."""
        result, s, n_y, n_x, sig, rank = _make_result()

        eager = result.get_decomposition_model()
        lazy_model = result.get_decomposition_model(lazy_output=True)

        assert isinstance(lazy_model.data, da.Array)
        np.testing.assert_allclose(
            lazy_model.data.compute(), eager.data, rtol=RTOL, atol=atol
        )

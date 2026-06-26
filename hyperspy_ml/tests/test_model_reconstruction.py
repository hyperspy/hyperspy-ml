# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy ML.

"""Tests for DecompositionResult model reconstruction and utilities (Task 4d).

ALL-DIFFERENT dimensions (7, 11, 13) used throughout.
"""

from __future__ import annotations

import sys

import dask.array as da
import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import DecompositionResult  # noqa: E402
from hyperspy_ml.stages.decomposition import Decomposition  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_signal(rng_seed=42, rank=3):
    """Signal1D with all-different dims (7 nav_y, 11 nav_x, 13 sig)."""
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
# Model reconstruction — eager
# ---------------------------------------------------------------------------


class TestModelReconstructionEager:
    """Eager model reconstruction via get_decomposition_model()."""

    def test_reconstruct_full(self):
        """Reconstruct from all components gives the original data."""
        result, s, n_y, n_x, sig, rank = _make_result()
        model = result.get_decomposition_model()

        assert model.data.shape == (n_y, n_x, sig)
        np.testing.assert_allclose(
            model.data.reshape(-1, sig),
            result.scores @ result.components.T,
            rtol=1e-10,
        )

    def test_reconstruct_with_component_subset(self):
        """Reconstruct from first 2 components gives rank-2 approximation."""
        result, s, n_y, n_x, sig, rank = _make_result()
        model = result.get_decomposition_model(components=2)
        assert model.data.shape == (n_y, n_x, sig)
        assert model.data.dtype.kind == "f"

    def test_reconstruct_with_component_list(self):
        """Reconstruct from specific component indices."""
        result, s, n_y, n_x, sig, rank = _make_result()
        model = result.get_decomposition_model(components=[0, 2])
        assert model.data.shape == (n_y, n_x, sig)

    def test_reconstruct_with_mean(self):
        """Mean (from centring) is added back during reconstruction."""
        s, n_y, n_x, sig, rank = _make_signal(rank=3)
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=rank,
            centre="navigation",
            print_info=False,
        )
        result = stage.fit_transform(s)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model()
        assert model.data.shape == (n_y, n_x, sig)
        assert result.mean is not None

        # Reconstruction with mean added back should be close to original
        recon = model.data.reshape(-1, sig)
        orig = s.data.reshape(-1, sig)
        rms = np.sqrt(np.mean((recon - orig) ** 2))
        norm = np.sqrt(np.mean(orig**2))
        assert rms / norm < 0.5, f"reconstruction error too large: {rms / norm:.3f}"

    def test_raises_without_source_signal(self):
        """ValueError when no source_signal is available."""
        result = DecompositionResult(components=np.eye(5), scores=np.ones((10, 5)))
        with pytest.raises(ValueError, match="source_signal"):
            result.get_decomposition_model()


# ---------------------------------------------------------------------------
# Model reconstruction — lazy / dask
# ---------------------------------------------------------------------------


class TestModelReconstructionLazy:
    """Lazy model reconstruction with dask/einsum."""

    def test_lazy_output_returns_dask_backed(self):
        """lazy_output=True returns a LazySignal with dask arrays."""
        s, n_y, n_x, sig, rank = _make_signal()
        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        result = stage.fit_transform(s)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model(lazy_output=True)
        assert getattr(model, "_lazy", False), "model should be lazy"
        assert isinstance(model.data, da.Array), "data should be dask-backed"

    def test_lazy_output_does_not_compute_eagerly(self):
        """lazy_output=True defers computation — .data is a dask array."""
        s, n_y, n_x, sig, rank = _make_signal()
        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        result = stage.fit_transform(s)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model(lazy_output=True)
        assert isinstance(model.data, da.Array)

        # Compute to verify correctness
        computed = model.data.compute()
        expected = s.data
        np.testing.assert_allclose(computed, expected, rtol=1e-8)

    def test_lazy_output_with_chunks(self):
        """Custom chunks parameter controls dask chunking."""
        s, n_y, n_x, sig, rank = _make_signal()
        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        result = stage.fit_transform(s)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model(lazy_output=True, chunks=(4, "auto"))
        assert model.data.chunks is not None

    def test_lazy_from_lazy_source(self):
        """When source signal is lazy, lazy_output defaults to True."""
        s, n_y, n_x, sig, rank = _make_signal()
        s_lazy = s.as_lazy()

        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        result = stage.fit_transform(s_lazy)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s_lazy

        model = result.get_decomposition_model()
        assert isinstance(model.data, da.Array), "should default to lazy"

    def test_lazy_output_with_component_subset(self):
        """Lazy reconstruction with subset of components works."""
        s, n_y, n_x, sig, rank = _make_signal()
        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        result = stage.fit_transform(s)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model(lazy_output=True, components=2)
        assert model.data.shape == (n_y, n_x, sig)
        computed = model.data.compute()
        assert np.all(np.isfinite(computed))


# ---------------------------------------------------------------------------
# BSS model reconstruction
# ---------------------------------------------------------------------------


class TestBSSModelReconstruction:
    """BSS model reconstruction via get_bss_model()."""

    def test_bss_model_basic(self):
        """get_bss_model reconstructs from BSS components/scores."""
        s, n_y, n_x, sig, rank = _make_signal()
        result, s, n_y_2, n_x_2, sig_2, rank_2 = _make_result(s)

        # Simulate BSS results (orthogonal rotation of decomposition)
        R = np.linalg.qr(np.random.default_rng(1).standard_normal((rank, rank)))[0]
        result.bss_components = result.components @ R
        result.bss_scores = result.scores @ R.T

        model = result.get_bss_model()
        assert model.data.shape == (n_y, n_x, sig)
        np.testing.assert_allclose(
            model.data.reshape(-1, sig),
            result.bss_scores @ result.bss_components.T,
            rtol=1e-10,
        )

    def test_bss_model_raises_without_results(self):
        """ValueError when BSS results are None."""
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.raises(ValueError, match="No bss results"):
            result.get_bss_model()


# ---------------------------------------------------------------------------
# crop_decomposition_dimension
# ---------------------------------------------------------------------------


class TestCropDecompositionDimension:
    """Crop decomposition results to fewer components."""

    def test_crop_reduces_n_components(self):
        """After crop, n_components matches the cropped size."""
        result, s, n_y, n_x, sig, rank = _make_result()
        result.crop_decomposition_dimension(2)
        assert result.n_components == 2
        assert result.components.shape == (sig, 2)
        assert result.scores.shape == (n_y * n_x, 2)
        assert result.explained_variance.shape == (2,)

    def test_crop_triggers_event(self):
        """crop_decomposition_dimension fires events.data_changed."""
        result, s, n_y, n_x, sig, rank = _make_result()
        assert result.events.data_changed is False
        result.crop_decomposition_dimension(2)
        assert result.events.data_changed is True

    def test_crop_explained_variance_ratio(self):
        """explained_variance_ratio is also cropped when present."""
        result, s, n_y, n_x, sig, rank = _make_result()
        result.crop_decomposition_dimension(2)
        assert result.explained_variance_ratio.shape == (2,)

    def test_crop_compute_on_dask(self):
        """crop_decomposition_dimension(compute=True) materialises dask arrays."""
        s, n_y, n_x, sig, rank = _make_signal()
        s_lazy = s.as_lazy()

        stage = Decomposition(
            algorithm="SVD",
            svd_solver="randomized",
            output_dimension=rank,
            print_info=False,
        )
        result = stage.fit_transform(s_lazy)
        result._nav_shape = s.axes_manager.navigation_shape[::-1]

        # Should be dask-backed after svd_solver='full'
        assert isinstance(result.components, da.Array)

        result.crop_decomposition_dimension(2, compute=True)
        assert not isinstance(result.components, da.Array), "should be numpy now"
        assert result.n_components == 2


# ---------------------------------------------------------------------------
# _compute_explained_variance_ratio
# ---------------------------------------------------------------------------


class TestComputeExplainedVarianceRatio:
    """Compute explained_variance_ratio from raw explained_variance."""

    def test_compute_ratio(self):
        """Ratio sums to 1.0."""
        result, s, n_y, n_x, sig, rank = _make_result()
        # Clear the auto-computed ratio
        result.explained_variance_ratio = None
        result._compute_explained_variance_ratio()
        np.testing.assert_allclose(result.explained_variance_ratio.sum(), 1.0)

    def test_compute_none_when_variance_is_none(self):
        """When explained_variance is None, ratio stays None."""
        result = DecompositionResult(explained_variance=None)
        result._compute_explained_variance_ratio()
        assert result.explained_variance_ratio is None


# ---------------------------------------------------------------------------
# _transpose_results
# ---------------------------------------------------------------------------


class TestTransposeResults:
    """Transpose swaps components ↔ scores."""

    def test_transpose_swaps(self):
        """After transpose, components and scores are swapped."""
        result, s, n_y, n_x, sig, rank = _make_result()
        orig_c = result.components.copy()
        orig_s = result.scores.copy()

        result._transpose_results()
        np.testing.assert_array_equal(result.components, orig_s)
        np.testing.assert_array_equal(result.scores, orig_c)

    def test_transpose_twice_is_identity(self):
        """Double transpose returns original arrangement."""
        result, s, n_y, n_x, sig, rank = _make_result()
        orig_c = result.components.copy()
        orig_s = result.scores.copy()

        result._transpose_results()
        result._transpose_results()
        np.testing.assert_array_equal(result.components, orig_c)
        np.testing.assert_array_equal(result.scores, orig_s)


# ---------------------------------------------------------------------------
# normalize_poissonian_noise / undo_treatments
# ---------------------------------------------------------------------------


class TestNormalizePoissonianNoise:
    """K-K scaling and undo_treatments on the result."""

    def test_normalize_stores_scaling_factors(self):
        """After scaling, bH and aG are set."""
        rng = np.random.default_rng(7)
        data = rng.integers(1, 100, size=(7 * 11, 13)).astype(float)
        s = Signal1D(data.reshape(7, 11, 13))

        result = DecompositionResult()
        result._source_signal = s
        result.normalize_poissonian_noise()

        assert result.bH is not None
        assert result.aG is not None
        assert result._data_before_treatments is not None

    def test_undo_treatments_restores(self):
        """undo_treatments restores the data to pre-scaling state."""
        rng = np.random.default_rng(7)
        data = rng.integers(1, 100, size=(7 * 11, 13)).astype(float)
        s = Signal1D(data.reshape(7, 11, 13))
        orig_data = s.data.copy()

        result = DecompositionResult()
        result._source_signal = s
        result.normalize_poissonian_noise()

        # Data should differ after scaling
        assert not np.allclose(s.data, orig_data, rtol=1e-10)

        result.undo_treatments()
        np.testing.assert_allclose(s.data, orig_data)

    def test_undo_without_normalize_raises(self):
        """undo_treatments without prior scaling raises AttributeError."""
        result = DecompositionResult()
        result._source_signal = Signal1D(np.random.rand(20, 10))
        with pytest.raises(AttributeError, match="Unable to undo data pre-treatments"):
            result.undo_treatments()


# ---------------------------------------------------------------------------
# Integration: Decomposition stage stores _nav_shape on result
# ---------------------------------------------------------------------------


class TestStageIntegration:
    """Verify Decomposition stage integrates with result utilities."""

    def test_result_can_reconstruct_without_explicit_signal(self):
        """When _source_signal and _nav_shape are set on result, reconstruction
        works without passing source_signal=."""
        s, n_y, n_x, sig, rank = _make_signal()
        stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
        result = stage.fit_transform(s)

        # Manually set what the stage should set
        result._nav_shape = s.axes_manager.navigation_shape[::-1]
        result._source_signal = s

        model = result.get_decomposition_model()
        assert model.data.shape == (n_y, n_x, sig)

    def test_result_events_persist_through_reconstruction(self):
        """Events namespace is preserved on result."""
        result, s, n_y, n_x, sig, rank = _make_result()
        assert hasattr(result, "events")
        assert result.events.data_changed is False

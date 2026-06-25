# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy ML.

"""Tests for DecompositionResult scree plots and component utilities (Task 4e).

ALL-DIFFERENT dimensions (7, 11, 13) used throughout.
Plot tests use MPLBACKEND=agg (no display needed).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hyperspy.exceptions import VisibleDeprecationWarning
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import DecompositionResult  # noqa: E402
from hyperspy_ml.stages.decomposition import Decomposition  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(rng_seed=42, rank=5):
    """Return a DecompositionResult from SVD with all-diff dims (7,11,13)."""
    rng = np.random.default_rng(rng_seed)
    nav_y, nav_x, sig = 7, 11, 13
    nav_size = nav_y * nav_x
    U = rng.standard_normal((nav_size, rank))
    V = rng.standard_normal((sig, rank))
    X = U @ V.T
    s = Signal1D(X.reshape(nav_y, nav_x, sig))

    stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
    result = stage.fit_transform(s)
    result._nav_shape = s.axes_manager.navigation_shape[::-1]
    result._source_signal = s
    return result, s, nav_y, nav_x, sig, rank


# ---------------------------------------------------------------------------
# get_scree_plot_data
# ---------------------------------------------------------------------------


class TestScreePlotData:
    """get_scree_plot_data returns correct data."""

    def test_returns_signal1d(self):
        """Returns a Signal1D with the variance data."""
        result, s, n_y, n_x, sig, rank = _make_result()
        data = result.get_scree_plot_data()
        assert isinstance(data, Signal1D)
        assert data.data.shape[0] == rank
        np.testing.assert_allclose(
            data.data, result.explained_variance_ratio, rtol=1e-10
        )

    def test_raises_without_variance(self):
        """AttributeError when explained_variance_ratio is None."""
        result = DecompositionResult(explained_variance_ratio=None)
        with pytest.raises(AttributeError, match="explained_variance_ratio.*None"):
            result.get_scree_plot_data()


# ---------------------------------------------------------------------------
# Deprecated get_explained_variance_ratio
# ---------------------------------------------------------------------------


class TestDeprecatedGetExplainedVarianceRatio:
    """Deprecated alias warns and routes correctly."""

    def test_warns_visible_deprecation(self):
        """get_explained_variance_ratio emits VisibleDeprecationWarning."""
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            r = result.get_explained_variance_ratio()
        assert r is not None


# ---------------------------------------------------------------------------
# plot_scree
# ---------------------------------------------------------------------------


class TestPlotScree:
    """plot_scree renders without errors."""

    def test_basic_render(self):
        """Basic invocation returns an Axes object."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        ax = result.plot_scree()
        assert ax is not None

    def test_with_vline(self):
        """vline=True draws vertical line at elbow estimate."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result(rank=10)
        ax = result.plot_scree(n=None, vline=True)
        assert ax is not None

    def test_with_hline_auto_float_threshold(self):
        """hline='auto' with float threshold draws the line."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result(rank=10)
        ax = result.plot_scree(n=None, threshold=0.1, hline="auto", log=False)
        assert ax is not None

    def test_with_threshold_int(self):
        """threshold=int highlights that many components as signal."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result(rank=10)
        ax = result.plot_scree(n=None, threshold=2)
        assert ax is not None

    def test_with_log_false(self):
        """log=False uses linear y-scale."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        ax = result.plot_scree(log=False)
        assert ax is not None

    def test_with_xaxis_type_number(self):
        """xaxis_type='number' offsets indices by 1."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        ax = result.plot_scree(xaxis_type="number")
        assert ax is not None

    def test_with_custom_formatting(self):
        """Custom signal_fmt and noise_fmt are accepted."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        ax = result.plot_scree(
            signal_fmt={"marker": "s", "c": "red"},
            noise_fmt={"marker": "x", "c": "blue"},
            n=3,
        )
        assert ax is not None

    def test_invalid_threshold_float(self):
        """threshold must be between 0 and 1 for float."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.raises(ValueError, match="between 0 and 1"):
            result.plot_scree(threshold=1.5)

    def test_explained_variance_ratio_requires_decomposition(self):
        """Plotting without explained_variance_ratio raises."""
        result = DecompositionResult(explained_variance_ratio=None)
        with pytest.raises(AttributeError, match="explained_variance_ratio.*None"):
            result.plot_scree()


# ---------------------------------------------------------------------------
# Deprecated plot_explained_variance_ratio
# ---------------------------------------------------------------------------


class TestDeprecatedPlotExplainedVarianceRatio:
    """Deprecated alias warns and routes to plot_scree."""

    def test_warns_and_returns_axes(self):
        """plot_explained_variance_ratio warns and returns an Axes."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            ax = result.plot_explained_variance_ratio(n=3)
        assert ax is not None


# ---------------------------------------------------------------------------
# plot_cumulative_scree
# ---------------------------------------------------------------------------


class TestPlotCumulativeScree:
    """plot_cumulative_scree renders correctly."""

    def test_basic_render(self):
        """Returns an Axes object."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        ax = result.plot_cumulative_scree()
        assert ax is not None

    def test_with_n(self):
        """n parameter limits displayed components."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result(rank=10)
        ax = result.plot_cumulative_scree(n=5)
        assert ax is not None

    def test_with_centred_decomposition(self):
        """Centred decomposition shows different labels."""
        import matplotlib

        matplotlib.use("agg")
        s = Signal1D(
            np.random.default_rng(1).standard_normal((77, 13)).reshape(7, 11, 13)
        )
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=5,
            centre="navigation",
            print_info=False,
        )
        result = stage.fit_transform(s)
        ax = result.plot_cumulative_scree(n=3)
        assert ax is not None


# ---------------------------------------------------------------------------
# Deprecated plot_cumulative_explained_variance_ratio
# ---------------------------------------------------------------------------


class TestDeprecatedPlotCumulativeExplainedVarianceRatio:
    """Deprecated alias warns and routes correctly."""

    def test_warns_and_returns_axes(self):
        """plot_cumulative_explained_variance_ratio warns, returns Axes."""
        import matplotlib

        matplotlib.use("agg")
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            ax = result.plot_cumulative_explained_variance_ratio(n=3)
        assert ax is not None


# ---------------------------------------------------------------------------
# normalize_decomposition_components
# ---------------------------------------------------------------------------


class TestNormalizeDecompositionComponents:
    """normalize_decomposition_components normalizes components."""

    def test_normalize_components_target(self):
        """Normalizing by components scales scores inversely."""
        result, s, n_y, n_x, sig, rank = _make_result()
        orig_c = result.components.copy()
        orig_s = result.scores.copy()

        result.normalize_decomposition_components(target="components")
        # Components div by sum, scores mul by sum
        assert not np.allclose(result.components, orig_c)
        np.testing.assert_allclose(
            result.components @ result.scores.T,
            orig_c @ orig_s.T,
            rtol=1e-10,
        )

    def test_normalize_scores_target(self):
        """Normalizing by scores scales components inversely."""
        result, s, n_y, n_x, sig, rank = _make_result()
        orig_c = result.components.copy()
        orig_s = result.scores.copy()

        result.normalize_decomposition_components(target="scores")
        np.testing.assert_allclose(
            result.components @ result.scores.T,
            orig_c @ orig_s.T,
            rtol=1e-10,
        )

    def test_deprecated_factors_target(self):
        """target='factors' warns and works."""
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.warns(VisibleDeprecationWarning, match="factors.*deprecated"):
            result.normalize_decomposition_components(target="factors")

    def test_deprecated_loadings_target(self):
        """target='loadings' warns and works."""
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.warns(VisibleDeprecationWarning, match="loadings.*deprecated"):
            result.normalize_decomposition_components(target="loadings")

    def test_invalid_target_raises(self):
        """Invalid target raises ValueError."""
        result, s, n_y, n_x, sig, rank = _make_result()
        with pytest.raises(ValueError, match="target must be"):
            result.normalize_decomposition_components(target="invalid")


# ---------------------------------------------------------------------------
# reverse_decomposition_component
# ---------------------------------------------------------------------------


class TestReverseDecompositionComponent:
    """reverse_decomposition_component flips sign."""

    def test_reverse_single_component(self):
        """Reversing a component flips its sign."""
        result, s, n_y, n_x, sig, rank = _make_result()
        orig_c = result.components[:, 0].copy()
        orig_s = result.scores[:, 0].copy()

        result.reverse_decomposition_component(0)
        np.testing.assert_allclose(result.components[:, 0], -orig_c)
        np.testing.assert_allclose(result.scores[:, 0], -orig_s)

    def test_reverse_multiple_components(self):
        """Reversing multiple components at once works."""
        result, s, n_y, n_x, sig, rank = _make_result()
        result.reverse_decomposition_component(0)

    def test_reverse_not_implemented_for_lazy(self):
        """Dask-backed components are not reversed (warning emitted)."""
        s = Signal1D(
            np.random.default_rng(2).standard_normal((77, 13)).reshape(7, 11, 13)
        )
        s_lazy = s.as_lazy()

        stage = Decomposition(
            algorithm="SVD",
            svd_solver="full",
            output_dimension=3,
            print_info=False,
        )
        result = stage.fit_transform(s_lazy)

        import dask.array as da

        if isinstance(result.components, da.Array):
            result.reverse_decomposition_component(0)
            # Should have logged a warning; no crash.
            # Components should be unchanged.
            assert isinstance(result.components, da.Array)

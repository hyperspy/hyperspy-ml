# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy ML.
#
# HyperSpy ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy ML. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Tests for the Decomposition stage class.

ALL-DIFFERENT dimensions (7, 11, 13) are used throughout so that axis
reversals are immediately visible.
"""

import importlib
import sys

import numpy as np
import pytest
from hyperspy import signals

# Ensure we load from the extract location, not another editable install.
if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.stages.decomposition import Decomposition  # noqa: E402
from hyperspy_ml.utils.preprocessing import estimate_elbow_position  # noqa: E402

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_low_rank_signal(rng_seed=42, rank=3):
    """Create a Signal1D with rank-*rank* structure and all-different dims.

    Shape: (nav_y=7, nav_x=11 | sig=13).
    """
    rng = np.random.default_rng(rng_seed)
    nav_size = 7 * 11  # 77
    sig_size = 13
    U = rng.normal(size=(sig_size, rank))
    V = rng.normal(size=(nav_size, rank))
    X = V @ U.T  # (77, 13)
    X_3d = X.reshape(7, 11, 13)
    return signals.Signal1D(X_3d)


# ---------------------------------------------------------------------------
# Unit tests — validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Verify that invalid inputs raise the expected errors."""

    def test_integer_dtype_rejected(self):
        """Decomposition requires float data."""
        s = signals.Signal1D(np.arange(7 * 11 * 13, dtype=np.int32).reshape(7, 11, 13))
        stage = Decomposition(print_info=False)
        with pytest.raises(TypeError, match="must be of the floating-point"):
            stage.fit(s)

    def test_navigation_size_too_small(self):
        """Single navigation pixel cannot be decomposed."""
        s = signals.Signal1D(np.random.rand(1, 13))
        stage = Decomposition(print_info=False)
        with pytest.raises(ValueError, match="navigation_size < 2"):
            stage.fit(s)

    def test_output_dimension_not_positive(self):
        """output_dimension must be a positive int."""
        s = _make_low_rank_signal()
        for bad_val in [-1, 0, 2.5, True]:
            stage = Decomposition(output_dimension=bad_val, print_info=False)
            with pytest.raises(ValueError, match="output_dimension"):
                stage.fit(s)

    def test_svd_solver_requires_output_dimension(self):
        """randomized / incremental solvers need output_dimension."""
        s = _make_low_rank_signal()
        stage = Decomposition(svd_solver="randomized", print_info=False)
        with pytest.raises(ValueError, match="output_dimension"):
            stage.fit(s)

    def test_invalid_centre_value(self):
        """Only None, 'navigation', 'signal' are accepted."""
        s = _make_low_rank_signal()
        stage = Decomposition(centre="invalid", print_info=False)
        with pytest.raises(ValueError, match="centre"):
            stage.fit(s)

    def test_invalid_reproject_value(self):
        """Only None, 'navigation', 'signal', 'both' are accepted."""
        s = _make_low_rank_signal()
        stage = Decomposition(reproject="invalid", print_info=False)
        with pytest.raises(ValueError, match="reproject"):
            stage.fit(s)

    def test_poissonian_noise_conflicts_with_centre(self):
        """K-K scaling requires centre=None."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            normalize_poissonian_noise=True, centre="navigation", print_info=False
        )
        with pytest.raises(ValueError, match="normalize_poissonian_noise"):
            stage.fit(s)

    def test_deferred_algorithms_raise(self):
        """MLPCA, RPCA, ORPCA, ORNMF are deferred."""
        s = _make_low_rank_signal()
        for algo in ["MLPCA", "RPCA", "ORPCA", "ORNMF"]:
            stage = Decomposition(algorithm=algo, output_dimension=3, print_info=False)
            with pytest.raises(NotImplementedError, match="deferred"):
                stage.fit_transform(s)

    def test_unknown_algorithm_raises(self):
        """Unknown string algorithms raise ValueError."""
        s = _make_low_rank_signal()
        stage = Decomposition(algorithm="nonexistent", print_info=False)
        with pytest.raises(ValueError, match="not recognised"):
            stage.fit_transform(s)


# ---------------------------------------------------------------------------
# SVD path tests
# ---------------------------------------------------------------------------


class TestSVD:
    """SVD decomposition via hyperspy_ml_algorithms.SVDPCA."""

    def test_basic_svd(self):
        """SVD returns correct shapes and a sensible reconstruction."""
        s = _make_low_rank_signal()
        stage = Decomposition(algorithm="SVD", output_dimension=3, print_info=False)
        result = stage.fit_transform(s)

        assert result.components.shape == (13, 3)
        assert result.scores.shape == (77, 3)
        assert result.explained_variance is not None
        assert result.explained_variance.shape == (3,)
        assert result.explained_variance_ratio is not None
        assert result.n_components == 3
        assert result.centre is None
        assert result.mean is None

        # Reconstruction should be close
        X_recon = result.scores @ result.components.T
        orig_flat = s.data.reshape(77, 13)
        recon_error = np.linalg.norm(X_recon - orig_flat)
        assert recon_error < 0.5 * np.linalg.norm(orig_flat)

    @pytest.mark.parametrize("centre", [None, "navigation", "signal"])
    def test_centre_modes(self, centre):
        """Each centre mode produces the correct mean shape."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, centre=centre, print_info=False
        )
        result = stage.fit_transform(s)

        if centre is None:
            assert result.mean is None
        elif centre == "navigation":
            assert result.mean.shape == (1, 13)
        elif centre == "signal":
            assert result.mean.shape == (77, 1)

    def test_svd_full_solver(self):
        """svd_solver='full' works with auto-determined output_dimension."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="SVD", output_dimension=None, svd_solver="full", print_info=False
        )
        result = stage.fit_transform(s)
        assert result.components.shape[1] == min(77, 13)
        assert result.scores.shape[1] == min(77, 13)

    def test_explained_variance_ratio_sum(self):
        """explained_variance_ratio sums to 1 (within tolerance)."""
        s = _make_low_rank_signal()
        stage = Decomposition(algorithm="SVD", output_dimension=3, print_info=False)
        result = stage.fit_transform(s)
        np.testing.assert_allclose(
            result.explained_variance_ratio.sum(), 1.0, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Mask handling tests
# ---------------------------------------------------------------------------


class TestMasks:
    """Verify navigation and signal masks work correctly."""

    def test_signal_mask(self):
        """Masking some signal channels excludes them from decomposition."""
        s = _make_low_rank_signal()
        # Mask last 3 signal channels
        sm = np.zeros(13, dtype=bool)
        sm[-3:] = True

        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            signal_mask=sm,
            print_info=False,
        )
        result = stage.fit_transform(s)
        # Components should be NaN in masked positions
        assert result.components.shape == (13, 3)
        assert np.all(np.isnan(result.components[-3:]))
        assert np.all(np.isfinite(result.components[:-3]))

    def test_navigation_mask(self):
        """Masking some navigation pixels excludes them from decomposition."""
        s = _make_low_rank_signal()
        nm = np.zeros(77, dtype=bool)
        nm[:5] = True  # Mask first 5 nav pixels

        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            navigation_mask=nm,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.scores.shape == (77, 3)
        assert np.all(np.isnan(result.scores[:5]))
        assert np.all(np.isfinite(result.scores[5:]))

    def test_both_masks(self):
        """Simultaneous nav and signal masking."""
        s = _make_low_rank_signal()
        nm = np.zeros(77, dtype=bool)
        nm[:3] = True
        sm = np.zeros(13, dtype=bool)
        sm[:2] = True

        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            navigation_mask=nm,
            signal_mask=sm,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert np.all(np.isnan(result.scores[:3]))
        assert np.all(np.isnan(result.components[:2]))

    def test_all_data_masked_raises(self):
        """Masking everything raises ValueError."""
        s = _make_low_rank_signal()
        sm = np.ones(13, dtype=bool)
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=1,
            signal_mask=sm,
            print_info=False,
        )
        with pytest.raises(ValueError, match="All the data are masked"):
            stage.fit_transform(s)


# ---------------------------------------------------------------------------
# Reprojection tests
# ---------------------------------------------------------------------------


class TestReprojection:
    """Verify reprojection modes recompute scores / components."""

    def test_reproject_none(self):
        """Default: no reprojection."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, reproject=None, print_info=False
        )
        result = stage.fit_transform(s)
        assert result.scores.shape == (77, 3)
        assert result.components.shape == (13, 3)

    def test_reproject_navigation(self):
        """Navigation reprojection recomputes scores over full data."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            reproject="navigation",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.scores.shape == (77, 3)
        assert np.all(np.isfinite(result.scores))

    def test_reproject_signal(self):
        """Signal reprojection recomputes components over full data."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            reproject="signal",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.components.shape == (13, 3)
        assert np.all(np.isfinite(result.components))

    def test_reproject_both(self):
        """Both reprojection modes together."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            reproject="both",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.scores.shape == (77, 3)
        assert result.components.shape == (13, 3)
        assert np.all(np.isfinite(result.scores))
        assert np.all(np.isfinite(result.components))


# ---------------------------------------------------------------------------
# sklearn paths
# ---------------------------------------------------------------------------


@skip_sklearn
class TestSklearn:
    """Test sklearn-backed decomposition algorithms."""

    def test_sklearn_pca(self):
        """PCA via sklearn.decomposition.PCA."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="sklearn_pca", output_dimension=3, print_info=False
        )
        result = stage.fit_transform(s)
        assert result.components.shape == (13, 3)
        assert result.scores.shape == (77, 3)
        # sklearn PCA always centres by default
        assert result.explained_variance is not None

    def test_sklearn_nmf(self):
        """NMF via sklearn.decomposition.NMF (needs positive data)."""
        s = _make_low_rank_signal()
        # NMF requires non-negative data
        s.data[:] = np.abs(s.data)

        stage = Decomposition(
            algorithm="NMF", output_dimension=3, max_iter=500, print_info=False
        )
        result = stage.fit_transform(s)
        assert result.components.shape == (13, 3)
        assert result.scores.shape == (77, 3)

    def test_custom_estimator(self):
        """Pass a sklearn PCA object directly as algorithm."""
        from sklearn.decomposition import PCA

        s = _make_low_rank_signal()
        estimator = PCA(n_components=3)
        stage = Decomposition(algorithm=estimator, print_info=False)
        result = stage.fit_transform(s)

        assert result.components.shape == (13, 3)
        assert result.scores.shape == (77, 3)
        assert result.algorithm.startswith("PCA(")

    def test_custom_estimator_params(self):
        """Custom estimator is available in result.params when return_info=True."""
        from sklearn.decomposition import PCA

        s = _make_low_rank_signal()
        estimator = PCA(n_components=3)
        stage = Decomposition(algorithm=estimator, return_info=True, print_info=False)
        result = stage.fit_transform(s)
        assert "estimator" in result.params


# ---------------------------------------------------------------------------
# DecompositionResult tests
# ---------------------------------------------------------------------------


class TestDecompositionResult:
    """Test the DecompositionResult container."""

    def test_defaults(self):
        """Dataclass has sensible defaults."""
        from hyperspy_ml.results.base import DecompositionResult

        r = DecompositionResult()
        assert r.components is None
        assert r.scores is None
        assert r.n_components == 0
        assert r.centre is None
        assert r.params == {}

    def test_params_dict(self):
        """Result stores the full parameter dict."""
        s = _make_low_rank_signal()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            centre="navigation",
            svd_solver="auto",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.params["algorithm"] == "SVD"
        assert result.params["output_dimension"] == 3
        assert result.params["centre"] == "navigation"
        assert result.params["svd_solver"] == "auto"


# ---------------------------------------------------------------------------
# Elbow position tests
# ---------------------------------------------------------------------------


class TestElbowPosition:
    """Verify the elbow estimator still works in the new location."""

    def test_elbow(self):
        variance = np.asarray(
            [10e-1, 5e-2, 9e-3, 1e-3, 9e-5, 5e-5, 3e-5, 2.2e-5, 1.9e-5, 1.8e-5]
        )
        elbow = estimate_elbow_position(variance)
        assert elbow == 4

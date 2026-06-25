# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Tests for hyperspy_ml.utils.preprocessing helpers.

Uses ALL-DIFFERENT dimensions (7, 11, 13, 17) so that axis reversals
are immediately visible.
"""

import dask.array as da
import numpy as np
import pytest

from hyperspy_ml.utils.preprocessing import (
    _keenan_kotula_scale,
    _nan_expand_rows,
    _normalize_components,
    _reproject_navigation_scores,
    _reproject_signal_components,
    _to_flat_bool,
    apply_preprocessing,
    center_data,
    estimate_elbow_position,
)

# ---------------------------------------------------------------------------
# _keenan_kotula_scale
# ---------------------------------------------------------------------------


class TestKeenanKotulaScale:
    """Verify K-K scaling with numpy + dask, with and without masks."""

    def test_keenan_kotula_no_masks_numpy(self):
        """K-K scaling with no masks on a numpy array — basic sanity."""
        data = np.arange(7 * 11 * 13).reshape(7, 11, 13).astype(float) + 1.0
        scaled, sqrt_aG, sqrt_bH = _keenan_kotula_scale(
            data, navigation_mask=None, signal_mask=None, ndim=2, sdim=1
        )
        # scaled must not modify masked positions (there are none here)
        assert scaled.shape == (7, 11, 13)
        assert sqrt_aG.shape == (7, 11)
        assert sqrt_bH.shape == (13,)

    def test_keenan_kotula_numpy_dask_bit_identical(self):
        """K-K scaling: numpy and dask paths produce bit-identical results."""
        rng = np.random.default_rng(42)
        data = rng.poisson(50, (7, 11, 13)).astype(float)
        nm = rng.random((7, 11)) < 0.1
        sm = rng.random(13) < 0.1

        # numpy path
        s_np, aG_np, bH_np = _keenan_kotula_scale(
            data, navigation_mask=nm, signal_mask=sm, ndim=2, sdim=1
        )

        # dask path
        d_data = da.from_array(data, chunks=(3, 5, 7))
        s_da, aG_da, bH_da = _keenan_kotula_scale(
            d_data, navigation_mask=nm, signal_mask=sm, ndim=2, sdim=1
        )

        np.testing.assert_allclose(s_da.compute(), s_np, rtol=1e-14)
        np.testing.assert_allclose(aG_da.compute(), aG_np, rtol=1e-14)
        np.testing.assert_allclose(bH_da.compute(), bH_np, rtol=1e-14)

    def test_keenan_kotula_no_masks_dask(self):
        """K-K scaling with no masks on a dask array."""
        data = da.random.default_rng(42).poisson(50, (7, 11, 13)).astype(float)
        scaled, sqrt_aG, sqrt_bH = _keenan_kotula_scale(
            data, navigation_mask=None, signal_mask=None, ndim=2, sdim=1
        )
        assert scaled.shape == (7, 11, 13)
        np.testing.assert_array_less(0, scaled.compute().min())  # all positive data
        assert sqrt_aG.compute().shape == (7, 11)
        assert sqrt_bH.compute().shape == (13,)

    def test_keenan_kotula_negative_raises(self):
        """K-K scaling raises ValueError for negative values in unmasked region."""
        data = np.ones((7, 11, 13))
        data[2, 3, 4] = -1.0
        with pytest.raises(ValueError, match="Negative values found"):
            _keenan_kotula_scale(
                data, navigation_mask=None, signal_mask=None, ndim=2, sdim=1
            )

    def test_keenan_kotula_all_masked_raises(self):
        """K-K scaling raises ValueError when all data are masked."""
        data = np.ones((7, 11, 13))
        full_nav_mask = np.ones((7, 11), dtype=bool)
        with pytest.raises(ValueError, match="All the data are masked"):
            _keenan_kotula_scale(
                data, navigation_mask=full_nav_mask, signal_mask=None, ndim=2, sdim=1
            )


# ---------------------------------------------------------------------------
# _nan_expand_rows
# ---------------------------------------------------------------------------


class TestNanExpandRows:
    """Verify NaN expansion for numpy and dask paths."""

    def test_nan_expand_numpy(self):
        """NaN expand populates unmasked positions and fills rest with NaN."""
        # 2 unmasked rows (positions 1,3 in mask) → arr must have 2 rows
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([True, False, True, False, True], dtype=bool)
        # total_rows=5, unmasked at indices 1,3 → arr rows 0,1
        result = _nan_expand_rows(arr, mask, total_rows=5)
        assert result.shape == (5, 2)
        np.testing.assert_array_equal(np.isnan(result[[0, 2, 4], :]), True)
        np.testing.assert_allclose(result[1], arr[0])
        np.testing.assert_allclose(result[3], arr[1])

    def test_nan_expand_dask(self):
        """NaN expand via dask produces same result as numpy path."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        mask = np.array([True, False, True, False, True], dtype=bool)

        expected = _nan_expand_rows(arr, mask, total_rows=5)

        d_arr = da.from_array(arr, chunks=(3, 2))
        result = _nan_expand_rows(d_arr, mask, total_rows=5)
        np.testing.assert_allclose(result.compute(), expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# _reproject_navigation_scores / _reproject_signal_components
# ---------------------------------------------------------------------------


class TestReprojection:
    """Verify reprojection helpers produce correct shapes and least-squares
    consistency."""

    def test_reproject_navigation_scores_shape(self):
        """Navigation reprojection returns (nav, k) loadings."""
        D = np.random.default_rng(1).random((7, 11))
        components = np.random.default_rng(2).random((11, 5))
        scores = _reproject_navigation_scores(D, components)
        assert scores.shape == (7, 5)

    def test_reproject_signal_components_shape(self):
        """Signal reprojection returns (sig, k) factors."""
        D = np.random.default_rng(3).random((7, 11))
        scores = np.random.default_rng(4).random((7, 5))
        factors = _reproject_signal_components(D, scores)
        assert factors.shape == (11, 5)

    def test_roundtrip_signal_reprojection_consistency(self):
        """Signal reprojection from nav scores is a rank-k approximation of D.

        _reproject_signal_components computes factors = (pinv(scores) @ D).T,
        which gives a rank-k approximation: D ≈ scores @ factors.T.
        """
        rng = np.random.default_rng(5)
        D = rng.random((7, 11))
        components = rng.random((11, 3))
        scores = _reproject_navigation_scores(D, components)
        reconstructed = _reproject_signal_components(D, scores)
        D_approx = scores @ reconstructed.T
        # The rank-k approximation from pseudoinverse should match or improve
        # upon the least-squares projection; verify it's reasonable.
        # Check that D_approx is closer to D than the mean prediction.
        residual_approx = np.linalg.norm(D - D_approx)
        residual_zero = np.linalg.norm(D)
        assert residual_approx < residual_zero


# ---------------------------------------------------------------------------
# _normalize_components
# ---------------------------------------------------------------------------


class TestNormalizeComponents:
    """Verify component normalisation with different functions."""

    def test_normalize_with_sum(self):
        """normalize_components with default np.sum scales correctly."""
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        other = np.array([[5.0, 6.0], [7.0, 8.0]])
        t_copy = target.copy()
        o_copy = other.copy()
        _normalize_components(target, other, function=np.sum)
        # after: target /= coeff, other *= coeff; coeff = sum(target_orig, axis=0)
        coeff = np.sum(t_copy, axis=0)  # [4, 6]
        np.testing.assert_allclose(target, t_copy / coeff)
        np.testing.assert_allclose(other, o_copy * coeff)


# ---------------------------------------------------------------------------
# _to_flat_bool
# ---------------------------------------------------------------------------


class TestToFlatBool:
    """Verify mask normalisation from various input types."""

    def test_none_returns_none(self):
        assert _to_flat_bool(None) is None

    def test_numpy_flat(self):
        result = _to_flat_bool(np.array([0, 1, 0, 1, 1]))
        np.testing.assert_array_equal(
            result, np.array([False, True, False, True, True])
        )

    def test_numpy_2d_ravels(self):
        result = _to_flat_bool(np.eye(3, dtype=bool))
        assert result.shape == (9,)
        np.testing.assert_array_equal(result[:4], [True, False, False, False])

    def test_dask_computes(self):
        mask = da.from_array(np.array([0, 1, 0, 1]), chunks=2)
        result = _to_flat_bool(mask)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([False, True, False, True]))


# ---------------------------------------------------------------------------
# centre_data
# ---------------------------------------------------------------------------


class TestCenterData:
    """Verify centering wrapper produces correct means and centred data."""

    # Use shape (7, 11, 13): ndim=2 nav, sdim=1 signal
    def test_center_none(self):
        data = np.random.default_rng(7).random((7, 11, 13))
        centred, mean = center_data(data, centre=None)
        np.testing.assert_array_equal(centred, data)
        assert mean is None

    def test_center_navigation(self):
        data = np.random.default_rng(8).random((7, 11, 13))
        centred, mean = center_data(data, centre="navigation", ndim=2)
        assert mean.shape == (1, 1, 13)
        np.testing.assert_allclose(mean.squeeze(), data.mean(axis=(0, 1)))
        # After adding back mean, get original
        np.testing.assert_allclose(centred + mean, data, rtol=1e-14)

    def test_center_signal(self):
        data = np.random.default_rng(9).random((7, 11, 13))
        centred, mean = center_data(data, centre="signal", ndim=2)
        assert mean.shape == (7, 11, 1)
        np.testing.assert_allclose(mean.squeeze(), data.mean(axis=-1))
        np.testing.assert_allclose(centred + mean, data, rtol=1e-14)

    def test_center_invalid_raises(self):
        with pytest.raises(ValueError, match="must be None"):
            center_data(np.ones((3, 4)), centre="invalid")


# ---------------------------------------------------------------------------
# apply_preprocessing
# ---------------------------------------------------------------------------


class TestApplyPreprocessing:
    """Verify the full preprocessing pipeline wrapper."""

    def test_apply_no_ops(self):
        data = np.random.default_rng(10).random((7, 11))
        processed, mean, aG, bH = apply_preprocessing(data)
        np.testing.assert_array_equal(processed, data)
        assert mean is None
        assert aG is None
        assert bH is None

    def test_apply_with_centering(self):
        data = np.random.default_rng(11).random((7, 11, 13))
        processed, mean, aG, bH = apply_preprocessing(
            data, centre="signal", ndim=2, sdim=1
        )
        assert processed.shape == (7, 11, 13)
        assert mean.shape == (7, 11, 1)
        assert aG is None
        assert bH is None

    def test_apply_kk_incompatible_with_centre(self):
        with pytest.raises(ValueError, match="only compatible"):
            apply_preprocessing(
                np.ones((7, 11)),
                centre="navigation",
                normalize_poissonian_noise=True,
            )

    def test_apply_kk_with_numpy(self):
        rng = np.random.default_rng(12)
        data = rng.poisson(50, (7, 11, 13)).astype(float)
        processed, mean, aG, bH = apply_preprocessing(
            data, normalize_poissonian_noise=True, ndim=2, sdim=1
        )
        assert processed.shape == (7, 11, 13)
        assert mean is None
        assert aG.shape == (7, 11)
        assert bH.shape == (13,)
        # Check K-K scales correctly: processed should differ from data
        assert not np.allclose(processed, data)


# ---------------------------------------------------------------------------
# estimate_elbow_position
# ---------------------------------------------------------------------------


class TestEstimateElbowPosition:
    """Verify elbow detection on known scree-plot curves."""

    def test_known_elbow_curve(self):
        """Elbow on [100, 50, 10, 5, 3, 2] → index 2 (value 10)."""
        curve = np.array([100.0, 50.0, 10.0, 5.0, 3.0, 2.0])
        pos = estimate_elbow_position(curve, log=True)
        assert pos == 2

    def test_short_curve(self):
        """Elbow on short curve yields valid index."""
        curve = np.array([1.0, 0.1])
        pos = estimate_elbow_position(curve, log=True)
        assert pos in (0, 1)

    def test_flat_curve(self):
        """Elbow on flat curve returns some index (no crash)."""
        curve = np.ones(10)
        pos = estimate_elbow_position(curve, log=True)
        assert 0 <= pos < 10

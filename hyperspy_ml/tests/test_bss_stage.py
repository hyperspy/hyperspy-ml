# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Tests for the BSS stage (Task 5).

ALL-DIFFERENT dimensions (7, 11, 13).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import DecompositionResult  # noqa: E402
from hyperspy_ml.stages.bss import BSS  # noqa: E402
from hyperspy_ml.stages.decomposition import Decomposition  # noqa: E402

skip_sklearn = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sklearn") is None,
    reason="sklearn not installed",
)


def _make_result(rng_seed=42, rank=6):
    """DecompositionResult with all-diff dims (7, 11, 13)."""
    rng = np.random.default_rng(rng_seed)
    nav, sig = 77, 13
    U = rng.standard_normal((nav, rank))
    V = rng.standard_normal((sig, rank))
    X = U @ V.T
    s = Signal1D(X.reshape(7, 11, sig))
    stage = Decomposition(algorithm="SVD", output_dimension=rank, print_info=False)
    result = stage.fit_transform(s)
    result._nav_shape = s.axes_manager.navigation_shape[::-1]
    return result


class TestBSSOrthomax:
    def test_orthomax_basic(self):
        """Orthomax runs and returns BSS components/scores."""
        lr = _make_result()
        bss = BSS(number_of_components=4, algorithm="orthomax", print_info=False)
        bss_result, _ = bss.fit_transform(lr)
        assert bss_result.bss_components.shape == (13, 4)
        assert bss_result.bss_scores.shape == (77, 4)
        assert bss_result.unmixing_matrix.shape == (4, 4)
        assert bss_result.bss_algorithm == "orthomax"

    def test_orthomax_on_scores(self):
        """BSS on scores instead of components."""
        lr = _make_result()
        bss = BSS(
            number_of_components=4,
            algorithm="orthomax",
            on_scores=True,
            print_info=False,
        )
        bss_result, _ = bss.fit_transform(lr)
        assert bss_result.bss_components.shape == (13, 4)
        assert bss_result.on_scores is True


@skip_sklearn
class TestBSSSklearnFastICA:
    def test_sklearn_fastica_basic(self):
        """sklearn FastICA runs and returns results."""
        lr = _make_result()
        bss = BSS(number_of_components=4, algorithm="sklearn_fastica", print_info=False)
        bss_result, _ = bss.fit_transform(lr)
        assert bss_result.bss_components.shape == (13, 4)
        assert bss_result.bss_scores.shape == (77, 4)

    def test_sklearn_fastica_return_info(self):
        """return_info=True returns the estimator."""
        lr = _make_result()
        bss = BSS(
            number_of_components=4,
            algorithm="sklearn_fastica",
            return_info=True,
            print_info=False,
        )
        _, info = bss.fit_transform(lr)
        assert info is not None


class TestBSSMDPGuard:
    def test_mdp_raises_valueerror(self):
        """MDP algorithms raise ImportError when mdp not installed."""
        import importlib

        if importlib.util.find_spec("mdp") is not None:
            pytest.skip("mdp is installed")
        lr = _make_result()
        bss = BSS(number_of_components=4, algorithm="JADE", print_info=False)
        with pytest.raises(ImportError, match="MDP"):
            bss.fit_transform(lr)


class TestBSSNormalizeReverse:
    def test_normalize_bss_components(self):
        """normalize_bss_components scales correctly."""
        lr = _make_result()
        bss = BSS(number_of_components=4, algorithm="orthomax", print_info=False)
        result, _ = bss.fit_transform(lr)
        orig_c = result.bss_components.copy()
        orig_s = result.bss_scores.copy()
        BSS.normalize_bss_components(result, target="components")
        assert not np.allclose(result.bss_components, orig_c)
        np.testing.assert_allclose(
            result.bss_components @ result.bss_scores.T,
            orig_c @ orig_s.T,
            rtol=1e-10,
        )

    def test_reverse_bss_component(self):
        """reverse_bss_component flips sign."""
        lr = _make_result()
        bss = BSS(number_of_components=4, algorithm="orthomax", print_info=False)
        result, _ = bss.fit_transform(lr)
        orig = result.bss_components[:, 0].copy()
        BSS.reverse_bss_component(result, 0)
        np.testing.assert_allclose(result.bss_components[:, 0], -orig)


class TestBSSValidation:
    def test_needs_decomposition(self):
        """BSS requires a prior decomposition."""
        lr = DecompositionResult()
        bss = BSS(number_of_components=3, print_info=False)
        with pytest.raises(AttributeError, match="decomposition must be performed"):
            bss.fit_transform(lr)

    def test_invalid_algorithm_raises(self):
        """Unknown algorithm raises ValueError."""
        lr = _make_result()
        bss = BSS(number_of_components=3, algorithm="nonexistent", print_info=False)
        with pytest.raises(ValueError, match="not recognised"):
            bss.fit_transform(lr)

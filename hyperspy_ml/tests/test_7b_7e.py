# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Tests for Task 7b-7e: LearningResults, I/O, Study, Pipeline, API."""

from __future__ import annotations

import sys
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import DecompositionResult
from hyperspy_ml.results.learning_results import LearningResults
from hyperspy_ml.stages.decomposition import Decomposition

zarr = pytest.importorskip("zarr", reason="zarr required")


def _make_legacy_npz(tmpdir):
    """Create a legacy .npz file with old-format keys."""
    path = f"{tmpdir}/legacy.npz"
    np.savez(
        path,
        factors=np.eye(5),
        loadings=np.ones((10, 5)),
        explained_variance=np.arange(5, dtype=float),
        output_dimension=5,
        decomposition_algorithm="SVD",
        centre=None,
        poissonian_noise_normalized=False,
    )
    return path


# ============================================================================
# 7b: LearningResults bridge
# ============================================================================


class TestLearningResultsBridge:
    def test_save_load_round_trip(self):
        lr = LearningResults()
        lr.components = np.eye(3)
        lr.scores = np.ones((10, 3))
        lr.decomposition_algorithm = "SVD"
        lr.output_dimension = 3

        with TemporaryDirectory() as tmp:
            path = f"{tmp}/test.npz"
            lr.save(path)
            lr2 = LearningResults()
            lr2.load(path)

        np.testing.assert_array_equal(lr2.components, lr.components)
        np.testing.assert_array_equal(lr2.scores, lr.scores)
        assert lr2.decomposition_algorithm == "SVD"

    def test_deprecated_property_warns(self):
        from hyperspy.exceptions import VisibleDeprecationWarning

        lr = LearningResults()
        lr.components = np.eye(3)
        with pytest.warns(VisibleDeprecationWarning, match="deprecated"):
            result = lr.factors
        np.testing.assert_array_equal(result, lr.components)

    def test_load_legacy_keys(self):
        """Loading .npz with old factors/loadings keys migrates correctly."""
        with TemporaryDirectory() as tmp:
            path = _make_legacy_npz(tmp)
            lr = LearningResults()
            lr.load(path)

        assert lr.components.shape == (5, 5)
        assert lr.scores.shape == (10, 5)
        assert lr.output_dimension == 5


# ============================================================================
# 7c: .hsml I/O with legacy .npz reader
# ============================================================================


class TestHSMLIOLegacy:
    def test_load_npz(self):
        from hyperspy_ml.results.io import load_npz

        with TemporaryDirectory() as tmp:
            path = _make_legacy_npz(tmp)
            result = load_npz(path)

        assert isinstance(result, DecompositionResult)
        np.testing.assert_array_equal(result.components, np.eye(5))
        assert result.algorithm == "SVD"

    def test_extract_results_from_signal(self):
        from hyperspy_ml.results.io import extract_results

        rng = np.random.default_rng(7)
        s = Signal1D(rng.standard_normal((20, 10)))
        s.decomposition(output_dimension=3)
        result = extract_results(s)
        assert result.components.shape == (10, 3)
        assert result.algorithm == "SVD"


# ============================================================================
# 7d: Study container
# ============================================================================


class TestStudy:
    def test_add_and_get(self):
        from hyperspy_ml.study.study import Study

        study = Study("test")
        rng = np.random.default_rng(1)
        s = Signal1D(rng.standard_normal((20, 10)))
        stage = Decomposition(output_dimension=2, print_info=False)
        result = stage.fit_transform(s)

        key = study.add(result, name="my_decomp")
        assert key == "my_decomp"
        assert "my_decomp" in study
        assert study["my_decomp"] is result

    def test_auto_name(self):
        from hyperspy_ml.study.study import Study

        study = Study()
        rng = np.random.default_rng(1)
        s = Signal1D(rng.standard_normal((20, 10)))
        stage = Decomposition(output_dimension=2, print_info=False)
        result = stage.fit_transform(s)

        key = study.add(result)
        assert key.startswith("DecompositionResult_")
        assert key in study

    def test_remove_dependents(self):
        from hyperspy_ml.study.study import Study

        study = Study()
        rng = np.random.default_rng(1)
        s = Signal1D(rng.standard_normal((20, 10)))
        stage = Decomposition(output_dimension=2, print_info=False)
        result = stage.fit_transform(s)

        study.add(result, "decom")
        study.add(result, "decom_bss")
        study.remove("decom")
        assert "decom" not in study
        assert "decom_bss" not in study

    def test_summary(self):
        from hyperspy_ml.study.study import Study

        study = Study("my_study")
        rng = np.random.default_rng(1)
        s = Signal1D(rng.standard_normal((20, 10)))
        stage = Decomposition(output_dimension=2, print_info=False)
        result = stage.fit_transform(s)
        study.add(result, "decomp")
        text = study.summary()
        assert "my_study" in text
        assert "DecompositionResult" in text

    def test_save_load_round_trip(self):
        from hyperspy_ml.study.study import Study

        study = Study("test_study")
        rng = np.random.default_rng(1)
        s = Signal1D(rng.standard_normal((20, 10)))
        stage = Decomposition(output_dimension=2, print_info=False)
        result = stage.fit_transform(s)
        study.add(result, "decomp")

        with TemporaryDirectory() as tmp:
            path = f"{tmp}/study.hsml"
            study.save(path)
            loaded = Study.load(path)

        assert "decomp" in loaded
        assert loaded.name == "test_study"

    def test_events(self):
        from hyperspy_ml.study.study import Study

        study = Study()
        events = []

        def on_add(key):
            events.append(("added", key))

        study.events.connect("result_added", on_add)
        rng = np.random.default_rng(1)
        s = Signal1D(rng.standard_normal((20, 10)))
        stage = Decomposition(output_dimension=2, print_info=False)
        result = stage.fit_transform(s)
        study.add(result, "my_key")
        assert events == [("added", "my_key")]


# ============================================================================
# 7d: Pipeline
# ============================================================================


class TestPipeline:
    def test_linear_run(self):
        from hyperspy_ml.pipeline.pipeline import Pipeline

        def step1(x):
            return x * 2

        def step2(x):
            return x + 1

        pipe = Pipeline(
            [
                ("double", step1, None),
                ("increment", step2, None),
            ]
        )
        result = pipe.run(3)
        assert result == 7

    def test_contract_validation(self):
        from hyperspy_ml.pipeline.pipeline import Pipeline

        def step_int(x):
            return x

        pipe = Pipeline([("int_step", step_int, (int,))])
        assert pipe.run(42) == 42
        with pytest.raises(TypeError, match="requires input type"):
            pipe.run("hello")

    def test_indexing(self):
        from hyperspy_ml.pipeline.pipeline import Pipeline

        def step1(x):
            return x * 2

        def step2(x):
            return x + 1

        pipe = Pipeline([("double", step1, None), ("increment", step2, None)])
        assert pipe[0] is step1
        sliced = pipe[1:2]
        assert len(sliced) == 1

    def test_partial_fit_dispatch(self):
        from hyperspy_ml.pipeline.pipeline import Pipeline

        class FakeStage:
            def partial_fit(self, data):
                return f"fit_{data}"

        pipe = Pipeline(
            [
                ("a", lambda x: x, None),
                ("b", FakeStage(), None),
            ]
        )
        result = pipe.partial_fit(42)
        assert result == "fit_42"


# ============================================================================
# 7e: Convenience API
# ============================================================================


class TestConvenienceAPI:
    def test_decompose_matches_stage(self):
        from hyperspy_ml.api import decompose

        rng = np.random.default_rng(42)
        s = Signal1D(rng.standard_normal((20, 10)))
        result = decompose(s, output_dimension=3)
        assert isinstance(result, DecompositionResult)
        assert result.components.shape == (10, 3)

        # Compare with direct stage path
        stage = Decomposition(output_dimension=3, print_info=False)
        direct = stage.fit_transform(s)
        np.testing.assert_allclose(result.explained_variance, direct.explained_variance)

    def test_extract_results_convenience(self):
        from hyperspy_ml.api import extract_results

        rng = np.random.default_rng(99)
        s = Signal1D(rng.standard_normal((20, 10)))
        s.decomposition(output_dimension=3)
        result = extract_results(s)
        assert result.components.shape == (10, 3)

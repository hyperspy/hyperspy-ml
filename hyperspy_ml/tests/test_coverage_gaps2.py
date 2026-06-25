# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Targeted tests to push stage-level coverage past 85%.

Covers BSS stage edge cases (mask validation, diff_order, comp_list),
decomposition stage alternative paths, and lazy decomposition paths.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from hyperspy.signals import Signal1D

if "/tmp/hyperspy-ml-extract" not in sys.path:
    sys.path.insert(0, "/tmp/hyperspy-ml-extract")

from hyperspy_ml.results.base import DecompositionResult
from hyperspy_ml.stages.bss import BSS
from hyperspy_ml.stages.clustering import Clustering
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


# ============================================================================
# Decomposition stage — alternative algorithm dispatch paths
# ============================================================================


class TestDecompositionDispatch:
    @skip_sklearn
    def test_sklearn_pca(self):
        """Decomposition with sklearn_pca."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="sklearn_pca", output_dimension=3, print_info=False
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3
        assert result.components.shape == (13, 3)

    @skip_sklearn
    def test_nmf(self):
        """Decomposition with NMF algorithm."""
        data = np.abs(np.random.default_rng(1).standard_normal((7, 11, 13)))
        s = Signal1D(data)
        stage = Decomposition(algorithm="NMF", output_dimension=3, print_info=False)
        result = stage.fit_transform(s)
        assert result.n_components == 3

    @skip_sklearn
    def test_sparse_pca(self):
        """Decomposition with sparse_pca."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="sparse_pca", output_dimension=2, print_info=False
        )
        result = stage.fit_transform(s)
        assert result.n_components == 2

    @skip_sklearn
    def test_orpca(self):
        """Decomposition with ORPCA algorithm."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(algorithm="ORPCA", output_dimension=3, print_info=False)
        result = stage.fit_transform(s)
        assert result.n_components == 3

    @skip_sklearn
    def test_ornmf(self):
        """Decomposition with ORNMF algorithm."""
        data = np.abs(np.random.default_rng(1).standard_normal((7, 11, 13)))
        s = Signal1D(data)
        stage = Decomposition(algorithm="ORNMF", output_dimension=3, print_info=False)
        result = stage.fit_transform(s)
        assert result.n_components == 3

    @skip_sklearn
    def test_return_info(self):
        """Decomposition with return_info=True."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, print_info=False, return_info=True
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_svd_with_specific_solver(self):
        """Decomposition with svd_solver='randomized'."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    @skip_sklearn
    def test_custom_estimator(self):
        """Decomposition with a custom estimator object."""
        from sklearn.decomposition import PCA

        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        estim = PCA(n_components=3)
        stage = Decomposition(algorithm=estim, output_dimension=None, print_info=False)
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_print_info(self):
        """Decomposition with print_info=True prints to stdout."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(algorithm="SVD", output_dimension=3, print_info=True)
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = stage.fit_transform(s)
        finally:
            sys.stdout = old_stdout
        assert result.n_components == 3


# ============================================================================
# Decomposition stage — mask and preprocessing paths
# ============================================================================


class TestDecompositionMaskPaths:
    def test_with_navigation_mask_numpy(self):
        """Decomposition with a numpy navigation mask."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        nm = np.zeros(77, dtype=bool)
        nm[:5] = True  # mask first 5 nav pixels
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, navigation_mask=nm, print_info=False
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_with_signal_mask_numpy(self):
        """Decomposition with a numpy signal mask."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        sm = np.zeros(13, dtype=bool)
        sm[0] = True
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, signal_mask=sm, print_info=False
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_with_both_masks_numpy(self):
        """Decomposition with both navigation and signal masks."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        nm = np.zeros(77, dtype=bool)
        nm[:3] = True
        sm = np.zeros(13, dtype=bool)
        sm[0] = True
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            navigation_mask=nm,
            signal_mask=sm,
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_with_centre_navigation(self):
        """Decomposition with centre='navigation'."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, centre="navigation", print_info=False
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3
        assert result.centre == "navigation"

    def test_with_centre_signal(self):
        """Decomposition with centre='signal'."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, centre="signal", print_info=False
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3
        assert result.centre == "signal"

    def test_get_safe_keep_mask_none(self):
        """_get_safe_keep_mask with None mask returns ones."""
        stage = Decomposition(algorithm="SVD", output_dimension=3)
        mask = stage._get_safe_keep_mask(None, 10)
        np.testing.assert_array_equal(mask, np.ones(10, dtype=bool))

    def test_get_safe_keep_mask_slice(self):
        """_get_safe_keep_mask with slice returns ones."""
        stage = Decomposition(algorithm="SVD", output_dimension=3)
        mask = stage._get_safe_keep_mask(slice(None), 10)
        np.testing.assert_array_equal(mask, np.ones(10, dtype=bool))

    def test_get_safe_keep_mask_array(self):
        """_get_safe_keep_mask with array returns complement."""
        stage = Decomposition(algorithm="SVD", output_dimension=3)
        nm = np.zeros(10, dtype=bool)
        nm[:3] = True
        result = stage._get_safe_keep_mask(nm, 10)
        assert result.sum() == 7  # 10 - 3

    def test_reproject_navigation(self):
        """Decomposition with reproject='navigation'."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            reproject="navigation",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_reproject_signal(self):
        """Decomposition with reproject='signal'."""
        s = Signal1D(np.random.default_rng(1).standard_normal((7, 11, 13)))
        stage = Decomposition(
            algorithm="SVD", output_dimension=3, reproject="signal", print_info=False
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3


# ============================================================================
# Lazy decomposition tests
# ============================================================================


@skip_dask
class TestLazyDecompositionPaths:
    def test_lazy_svd_randomized(self):
        """Lazy signal decomposition with SVD randomized."""
        import dask.array as da

        data = da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        s = Signal1D(data.reshape(7, 11, 13)).as_lazy()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_lazy_with_centre(self):
        """Lazy signal decomposition with centering."""
        import dask.array as da

        data = da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        s = Signal1D(data.reshape(7, 11, 13)).as_lazy()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            centre="navigation",
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_lazy_with_navigation_mask(self):
        """Lazy signal decomposition with navigation mask."""
        import dask.array as da

        data = da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        s = Signal1D(data.reshape(7, 11, 13)).as_lazy()
        nm = np.zeros(77, dtype=bool)
        nm[:3] = True
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            navigation_mask=nm,
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_lazy_with_signal_mask(self):
        """Lazy signal decomposition with signal mask."""
        import dask.array as da

        data = da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        s = Signal1D(data.reshape(7, 11, 13)).as_lazy()
        sm = np.zeros(13, dtype=bool)
        sm[0] = True
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            signal_mask=sm,
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_lazy_with_keenan_kotula(self):
        """Lazy signal decomposition with poisson noise normalization."""
        import dask.array as da

        data = da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        data_abs = da.abs(data) + 1.0
        s = Signal1D(data_abs.reshape(7, 11, 13)).as_lazy()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            normalize_poissonian_noise=True,
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_lazy_with_reproject(self):
        """Lazy signal decomposition with reproject='both'."""
        import dask.array as da

        data = da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        s = Signal1D(data.reshape(7, 11, 13)).as_lazy()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            reproject="both",
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3

    def test_lazy_non_dask_native_algorithm(self):
        """Lazy signal cascades to eager for non-dask algorithm."""
        import dask.array as da

        data = da.abs(
            da.random.default_rng(1).standard_normal((77, 13), chunks=(20, 13))
        )
        s = Signal1D(data.reshape(7, 11, 13)).as_lazy()
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            svd_solver="randomized",
            print_info=False,
        )
        result = stage.fit_transform(s)
        assert result.n_components == 3


# ============================================================================
# BSS stage — extra paths
# ============================================================================


class TestBSSStageExtra:
    @skip_sklearn
    def test_orthomax_on_scores(self):
        """BSS with on_scores=True."""
        result, s = _make_decom_result(rank=4)
        stage = BSS(
            number_of_components=3,
            algorithm="orthomax",
            on_scores=True,
            print_info=False,
        )
        bss_result, _extra = stage.fit_transform(result)
        assert bss_result.bss_components is not None

    @skip_sklearn
    def test_orthomax_with_comp_list(self):
        """BSS with explicit comp_list instead of number_of_components."""
        result, s = _make_decom_result(rank=4)
        stage = BSS(comp_list=[0, 2, 3], algorithm="orthomax", print_info=False)
        bss_result, _extra = stage.fit_transform(result)
        assert bss_result.bss_components is not None

    @skip_sklearn
    def test_orthomax_with_diff_order(self):
        """BSS with diff_order > 0."""
        result, s = _make_decom_result(rank=4)
        stage = BSS(
            number_of_components=3, algorithm="orthomax", diff_order=1, print_info=False
        )
        bss_result, _extra = stage.fit_transform(result)
        assert bss_result.bss_components is not None

    @skip_sklearn
    def test_orthomax_with_whiten_method(self):
        """BSS with whiten_method='PCA'."""
        result, s = _make_decom_result(rank=4)
        stage = BSS(
            number_of_components=3,
            algorithm="orthomax",
            whiten_method="PCA",
            print_info=False,
        )
        bss_result, _extra = stage.fit_transform(result)
        assert bss_result.bss_components is not None

    def test_build_params_dict(self):
        """Decomposition._build_params_dict includes extra_kwargs."""
        stage = Decomposition(
            algorithm="SVD",
            output_dimension=3,
            return_info=True,
            print_info=False,
            extra_param=42,
        )
        params = stage._build_params_dict(estim="dummy")
        assert params["algorithm"] == "SVD"
        assert "extra_param" in params

    def test_normalise_navigation_mask_basesignal(self):
        """_normalise_navigation_mask handles BaseSignal input."""
        nm_sig = Signal1D(np.zeros((7, 11)))
        stage = Decomposition(algorithm="SVD", output_dimension=3)
        result = stage._normalise_navigation_mask(nm_sig)
        assert result is not None

    def test_normalise_navigation_mask_numpy(self):
        """_normalise_navigation_mask handles numpy array (transposed)."""
        nm = np.zeros((11, 7))  # transpose aware
        stage = Decomposition(algorithm="SVD", output_dimension=3)
        result = stage._normalise_navigation_mask(nm)
        assert result is not None

    def test_normalise_navigation_mask_none(self):
        """_normalise_navigation_mask with None returns None."""
        stage = Decomposition(algorithm="SVD", output_dimension=3)
        result = stage._normalise_navigation_mask(None)
        assert result is None


# ============================================================================
# Clustering stage — extra paths
# ============================================================================


@skip_sklearn
class TestClusteringStageExtra:
    def test_minibatch_kmeans(self):
        """Clustering with minibatchkmeans."""
        result, s = _make_decom_result(rank=4)
        stage = Clustering(n_clusters=2, algorithm="minibatchkmeans", print_info=False)
        cr = stage.fit_transform(result, signal=s)
        assert cr.number_of_clusters == 2

    def test_agglomerative(self):
        """Clustering with agglomerative."""
        result, s = _make_decom_result(rank=4)
        stage = Clustering(n_clusters=2, algorithm="agglomerative", print_info=False)
        cr = stage.fit_transform(result, signal=s)
        assert cr.number_of_clusters == 2

    def test_preprocessing_standard(self):
        """Clustering with preprocessing='standard'."""
        result, s = _make_decom_result(rank=4)
        stage = Clustering(
            n_clusters=2,
            algorithm="kmeans",
            preprocessing="standard",
            print_info=False,
        )
        cr = stage.fit_transform(result, signal=s)
        assert cr.number_of_clusters == 2

    def test_preprocessing_norm(self):
        """Clustering with preprocessing='norm'."""
        result, s = _make_decom_result(rank=4)
        stage = Clustering(
            n_clusters=2,
            algorithm="kmeans",
            preprocessing="norm",
            print_info=False,
        )
        cr = stage.fit_transform(result, signal=s)
        assert cr.number_of_clusters == 2

    def test_cluster_source_signal(self):
        """Clustering with cluster_source='signal'."""
        result, s = _make_decom_result(rank=4)
        stage = Clustering(
            n_clusters=2,
            cluster_source="signal",
            algorithm="kmeans",
            print_info=False,
        )
        cr = stage.fit_transform(result, signal=s)
        assert cr.number_of_clusters == 2

    def test_algorithm_none_auto_selects(self):
        """Clustering with algorithm=None auto-selects KMeans."""
        result, s = _make_decom_result(rank=4)
        stage = Clustering(n_clusters=2, algorithm=None, print_info=False)
        cr = stage.fit_transform(result, signal=s)
        assert cr.number_of_clusters == 2

    def test_gap_statistic_estimation(self):
        """Clustering with gap statistic to estimate optimal k."""
        result, s = _make_decom_result(rank=6)
        stage = Clustering(
            n_clusters=2,
            algorithm="kmeans",
            print_info=False,
        )
        cr = stage.fit_transform(result, signal=s)
        assert cr.number_of_clusters == 2


# ============================================================================
# Study — additional paths
# ============================================================================


class TestStudyExtra:
    def test_study_add_with_params(self):
        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study.add(result, name="r1", source_signal=s, params={"key": "val"})
        assert study.get("r1") is not None

    def test_study_save_no_params(self):
        from tempfile import TemporaryDirectory

        from hyperspy_ml.study.study import Study

        study = Study(name="test_study")
        result, s = _make_decom_result(rank=3)
        study.add(result, name="r1")

        with TemporaryDirectory() as tmp:
            study.save(tmp + "/study_test")
            import zarr

            store = zarr.open(tmp + "/study_test", mode="r")
            assert store.attrs["name"] == "test_study"


# ============================================================================
# Utils preprocessing coverage
# ============================================================================


class TestPreprocessingExtra:
    def test_to_flat_bool_int_array(self):
        from hyperspy_ml.utils.preprocessing import _to_flat_bool

        arr = np.array([0, 1, 0, 1], dtype=int)
        result = _to_flat_bool(arr)
        np.testing.assert_array_equal(result, np.array([False, True, False, True]))

    def test_center_data_navigation(self):
        from hyperspy_ml.utils.preprocessing import center_data

        data = np.random.default_rng(1).standard_normal((20, 10))
        centered, mean = center_data(data, centre="navigation", ndim=1)
        assert centered.shape == data.shape
        assert mean is not None

    def test_center_data_signal(self):
        from hyperspy_ml.utils.preprocessing import center_data

        data = np.random.default_rng(1).standard_normal((20, 10))
        centered, mean = center_data(data, centre="signal", ndim=1)
        assert centered.shape == data.shape
        assert mean is not None

    def test_center_data_none(self):
        from hyperspy_ml.utils.preprocessing import center_data

        data = np.random.default_rng(1).standard_normal((20, 10))
        centered, mean = center_data(data, centre=None)
        assert centered is data
        assert mean is None


# ============================================================================
# results/base.py extra paths
# ============================================================================


class TestResultsBaseExtra:
    def test_decomposition_summary_method(self):
        """DecompositionResult.summary() returns string."""
        result, s = _make_decom_result(rank=3)
        result.algorithm = "SVD"
        summary = result.summary()
        assert "SVD" in summary

    def test_decomposition_result_repr(self):
        """DecompositionResult.__repr__ delegates to summary."""
        result, s = _make_decom_result(rank=3)
        r = repr(result)
        assert "Decomposition" in r

    def test_get_components_no_index(self):
        """get_components without idx returns all."""
        result, s = _make_decom_result(rank=3)
        c = result.get_components()
        assert c.shape == (13, 3)

    def test_get_scores_no_index(self):
        """get_scores without idx returns all."""
        result, s = _make_decom_result(rank=3)
        sc = result.get_scores()
        assert sc.shape == (77, 3)


# ============================================================================
# results/io.py extra paths
# ============================================================================


class TestResultsIOExtra:
    def test_save_result_with_source_signal(self):
        from tempfile import TemporaryDirectory

        from hyperspy_ml.results.base import save_result

        result, s = _make_decom_result(rank=3)
        with TemporaryDirectory() as tmp:
            save_result(result, tmp + "/test.hsml", source_signal=s)
            import zarr

            store = zarr.open(tmp + "/test.hsml", mode="r")
            assert "source_hash" in store.attrs

    def test_extract_results_from_signal(self):
        from hyperspy_ml.results.io import extract_results

        result, s = _make_decom_result(rank=3)
        result.write_to_signal(s)
        extracted = extract_results(s)
        assert isinstance(extracted, DecompositionResult)

    def test_load_npz(self):
        from tempfile import TemporaryDirectory

        from hyperspy_ml.results.io import load_npz

        result, s = _make_decom_result(rank=3)
        with TemporaryDirectory() as tmp:
            path = tmp + "/test.npz"
            lr = s.learning_results
            lr.components = result.components
            lr.scores = result.scores
            lr.decomposition_algorithm = "SVD"
            lr.output_dimension = 3
            lr.save(path)
            loaded = load_npz(path)
            assert isinstance(loaded, DecompositionResult)

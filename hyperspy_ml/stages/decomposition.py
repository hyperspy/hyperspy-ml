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

"""Matrix decomposition stage — the core of the MVA pipeline.

Replaces the inline algorithm dispatch inside :meth:`MVA.decomposition`
with a standalone, testable stage class that accepts a HyperSpy signal
and returns a :class:`~hyperspy_ml.results.DecompositionResult`.

Supported algorithms (common paths — fast track):
  * ``"SVD"`` — via :class:`hyperspy_ml_algorithms.SVDPCA`
  * ``"sklearn_pca"`` — via :class:`sklearn.decomposition.PCA`
  * ``"NMF"`` — via :class:`sklearn.decomposition.NMF`
  * ``"sparse_pca"`` / ``"mini_batch_sparse_pca"`` — via sklearn
  * Custom objects implementing ``fit_transform()`` or ``fit()+transform()``

Deferred paths (Task 4c):
  * ``"MLPCA"``, ``"RPCA"``, ``"ORPCA"``, ``"ORNMF"``
"""

from __future__ import annotations

import importlib
import logging
import warnings
from functools import partial
from typing import Any

import numpy as np

from hyperspy_ml.results.base import DecompositionResult
from hyperspy_ml.utils.preprocessing import (
    _nan_expand_rows,
    _reproject_navigation_scores,
    _reproject_signal_components,
    _to_flat_bool,
    apply_preprocessing,
)

_logger = logging.getLogger(__name__)

SKLEARN_INSTALLED = importlib.util.find_spec("sklearn") is not None

# Names of algorithms that delegate to sklearn.
_SKLEARN_ALGORITHM_NAMES = frozenset(
    {"sklearn_pca", "NMF", "sparse_pca", "mini_batch_sparse_pca"}
)

# Algorithms whose full-path implementation is deferred to a later task.
_DEFERRED_ALGORITHMS = frozenset({"MLPCA", "RPCA", "ORPCA", "ORNMF"})

# ---------------------------------------------------------------------------
# Scikit-learn algorithm factory
# ---------------------------------------------------------------------------


def _make_sklearn_estimator(
    algorithm: str, output_dimension: int | None, **kwargs: Any
):
    """Instantiate a sklearn decomposition estimator by its HyperSpy alias.

    Parameters
    ----------
    algorithm : str
        Must be a key of :data:`hyperspy_ml.utils.preprocessing.decomposition_algorithms`.
    output_dimension : int or None
        Passed as ``n_components`` to the sklearn constructor.
    **kwargs
        Forwarded to the sklearn constructor.

    Returns
    -------
    sklearn.base.BaseEstimator
    """
    from hyperspy_ml.utils.preprocessing import (
        _get_sklearn_algorithms,
    )

    klass = _get_sklearn_algorithms(algorithm)
    return klass(n_components=output_dimension, **kwargs)


# ---------------------------------------------------------------------------
# Decomposition stage
# ---------------------------------------------------------------------------


class Decomposition:
    """Standalone matrix-decomposition stage for HyperSpy signals.

    Parameters
    ----------
    normalize_poissonian_noise : bool, default False
        Apply Keenan-Kotula Poisson-noise scaling before decomposition.
        Only compatible with ``centre=None``.
    algorithm : str or object, default ``"SVD"``
        Algorithm to use.  String aliases include ``"SVD"``,
        ``"sklearn_pca"``, ``"NMF"``, ``"sparse_pca"``,
        ``"mini_batch_sparse_pca"``.  ``"MLPCA"``, ``"RPCA"``,
        ``"ORPCA"``, ``"ORNMF"`` are deferred to Task 4c.
        May also be an object that implements ``fit_transform()`` or
        ``fit()`` + ``transform()``.
    output_dimension : int or None, default None
        Number of components to retain.  ``None`` (full rank) is only
        valid for ``algebra='SVD'`` with ``svd_solver='full'``.
    centre : {None, ``"navigation"``, ``"signal"``}, default None
        Centering mode.  ``"navigation"`` subtracts the mean over nav.
        pixels; ``"signal"`` subtracts the mean over signal channels.
    auto_transpose : bool, default True
        If ``True``, automatically transpose data for performance.
        Only used by the ``"SVD"`` algorithm.
    navigation_mask : BaseSignal, ndarray, or None, default None
        Boolean mask where ``True`` marks excluded navigation pixels.
    signal_mask : BaseSignal, ndarray, or None, default None
        Boolean mask where ``True`` marks excluded signal channels.
    var_array : ndarray or None, default None
        Variance array for MLPCA (deferred).
    var_func : callable, ndarray, or None, default None
        Variance function for MLPCA (deferred).
    reproject : {None, ``"navigation"``, ``"signal"``, ``"both"``}, default None
        Reprojection mode.  Recomputes scores/components on the full
        (unmasked) data after a masked decomposition.
    return_info : bool, default False
        If ``True``, include the fitted estimator object in the result's
        ``params`` dict under key ``"estimator"``.
    print_info : bool, default True
        Print decomposition summary to stdout.
    svd_solver : {``"auto"``, ``"full"``, ``"arpack"``, ``"randomized"``}, default ``"auto"``
        SVD solver to use.  Only used for ``algorithm="SVD"``.
    num_chunks : int or None, default None
        Number of dask chunks to pass to the decomposition model at a time
        (lazy signals only).  More chunks require more memory but should
        run faster.  Not used for ``algorithm='SVD'`` with
        ``svd_solver='randomized'``.
    copy : bool, default False
        **Deprecated.** Data modifications are reversed mathematically
        after decomposition, so explicit copying is no longer needed.
    **kwargs
        Additional keyword arguments forwarded to the underlying algorithm.
    """

    def __init__(
        self,
        normalize_poissonian_noise: bool = False,
        algorithm: str | object = "SVD",
        output_dimension: int | None = None,
        centre: str | None = None,
        auto_transpose: bool = True,
        navigation_mask: Any = None,
        signal_mask: Any = None,
        var_array: np.ndarray | None = None,
        var_func: Any = None,
        reproject: str | None = None,
        return_info: bool = False,
        print_info: bool = True,
        svd_solver: str = "auto",
        num_chunks: int | None = None,
        copy: bool = False,
        **kwargs: Any,
    ) -> None:
        self.normalize_poissonian_noise = normalize_poissonian_noise
        self.algorithm = algorithm
        self.output_dimension = output_dimension
        self.centre = centre
        self.auto_transpose = auto_transpose
        self.navigation_mask = navigation_mask
        self.signal_mask = signal_mask
        self.var_array = var_array
        self.var_func = var_func
        self.reproject = reproject
        self.return_info = return_info
        self.print_info = print_info
        self.svd_solver = svd_solver
        self.num_chunks = num_chunks
        self.copy = copy
        self._extra_kwargs = kwargs

        # Fitted state.
        self._signal = None
        self._unfolded = False
        self._original_shape = None
        self._data_was_transposed = False

        # Stored for in-place reversal when copy=False.
        self._root_aG = None
        self._root_bH = None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def fit(self, signal) -> Decomposition:
        """Store the signal reference and validate input parameters.

        Parameters
        ----------
        signal : BaseSignal
            The signal to decompose.  Must have ``navigation_size >= 2``
            and float-typed data.

        Returns
        -------
        Decomposition
            Self, for chaining.
        """
        if self.copy:
            warnings.warn(
                "The `copy` parameter is deprecated and will be removed "
                "in a future release.  Data modifications are reversed "
                "mathematically after decomposition.",
                UserWarning,
                stacklevel=2,
            )
        self._validate_inputs(signal)
        self._signal = signal
        return self

    def fit_transform(self, signal) -> DecompositionResult:
        """Run decomposition on *signal* and return a result object.

        Parameters
        ----------
        signal : BaseSignal
            The signal to decompose.

        Returns
        -------
        DecompositionResult
            Typed container with ``components``, ``scores``,
            ``explained_variance``, ``explained_variance_ratio``,
            ``mean``, and other attributes.
        """
        self.fit(signal)
        return self._run()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self, signal) -> None:
        """Check all inputs before running the decomposition."""
        data = signal.data

        # Type guard
        if data.dtype.char not in np.typecodes["AllFloat"]:
            raise TypeError(
                "To perform a decomposition the data must be of the "
                f"floating-point (including complex) type, but the "
                f"current type is '{data.dtype}'.  Use change_dtype() "
                "to convert (e.g. s.change_dtype('float64'))."
            )

        if signal.axes_manager.navigation_size < 2:
            raise ValueError(
                "It is not possible to decompose a dataset with navigation_size < 2"
            )

        if self.output_dimension is not None:
            if not isinstance(self.output_dimension, (int, np.integer)) or isinstance(
                self.output_dimension, bool
            ):
                raise ValueError(
                    "`output_dimension` must be a positive integer, "
                    f"not {self.output_dimension!r}."
                )
            if self.output_dimension <= 0:
                raise ValueError(
                    "`output_dimension` must be a positive integer, "
                    f"got {self.output_dimension}."
                )

        if (
            self.svd_solver in ("randomized", "incremental")
            and self.output_dimension is None
        ):
            raise ValueError(
                "`output_dimension` must be specified when using "
                f"algorithm='SVD' with svd_solver={self.svd_solver!r}."
            )

        if self.centre not in (None, "navigation", "signal"):
            raise ValueError(
                f"`centre` must be None, 'navigation' or 'signal', not {self.centre!r}"
            )

        if self.reproject not in (None, "navigation", "signal", "both"):
            raise ValueError(
                "`reproject` must be None, 'navigation', 'signal' or "
                f"'both', not {self.reproject!r}"
            )

        if bool(self.normalize_poissonian_noise) and self.centre is not None:
            raise ValueError(
                "normalize_poissonian_noise=True is only compatible "
                f"with `centre=None`, not `centre={self.centre}`."
            )

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run(self) -> DecompositionResult:
        """Orchestrate the full decomposition pipeline."""
        signal = self._signal

        # Branch: lazy signals use a dask-native pipeline.
        if getattr(signal, "_lazy", False):
            return self._run_lazy()

        algorithm = self.algorithm
        to_print_lines: list[str] = [
            "Decomposition info:",
            f"  normalize_poissonian_noise={self.normalize_poissonian_noise}",
            f"  algorithm={algorithm}",
            f"  output_dimension={self.output_dimension}",
            f"  centre={self.centre}",
        ]

        # -------------------------------------------------------------
        # 1. Unfold multi-dimensional data into a 2-D matrix.
        # -------------------------------------------------------------
        self._original_shape = signal.data.shape
        self._unfolded = signal.unfold()
        try:
            # ---------------------------------------------------------
            # 2. Normalise masks to flat bool arrays.
            # ---------------------------------------------------------
            nm = navigation_mask = self._normalise_navigation_mask(self.navigation_mask)
            sm = signal_mask = _to_flat_bool(self.signal_mask)

            # ---------------------------------------------------------
            # 3. Transpose data if the first axis is not navigation.
            # ---------------------------------------------------------
            if signal.axes_manager[0].index_in_array == 0:
                dc = signal.data
                self._data_was_transposed = False
            else:
                dc = signal.data.T
                self._data_was_transposed = True

            # ---------------------------------------------------------
            # 4. Preprocess (Poisson-noise scaling + centering).
            # ---------------------------------------------------------
            sdim = signal.axes_manager.signal_dimension
            ndim = signal.axes_manager.navigation_dimension
            dc, mean, sqrt_aG, sqrt_bH = apply_preprocessing(
                dc,
                centre=self.centre,
                navigation_mask=nm,
                signal_mask=sm,
                normalize_poissonian_noise=self.normalize_poissonian_noise,
                ndim=ndim,
                sdim=sdim,
            )
            self._root_aG = sqrt_aG
            self._root_bH = sqrt_bH

            # ---------------------------------------------------------
            # 5. Convert None masks to slices and keep only
            #    unmasked rows / columns.
            # ---------------------------------------------------------
            # Matching the original MVA.decomposition convention:
            # when a mask is None it becomes slice(None), so
            # isinstance(mask, slice) is True and NaN-expand
            # blocks are skipped later on.
            if navigation_mask is None:
                navigation_mask = slice(None)
            if signal_mask is None:
                signal_mask = slice(None)

            _nm = (
                navigation_mask
                if isinstance(navigation_mask, slice)
                else ~navigation_mask
            )
            _sm = signal_mask if isinstance(signal_mask, slice) else ~signal_mask

            data_ = dc[:, _sm][_nm, :]
            if data_.size == 0:
                raise ValueError("All the data are masked, change the mask.")

            # ---------------------------------------------------------
            # 6. Dispatch to the chosen algorithm.
            # ---------------------------------------------------------
            (
                factors,
                loadings,
                explained_variance,
                estim,
            ) = self._dispatch_algorithm(data_, to_print_lines)

            # ---------------------------------------------------------
            # 7. Compute explained-variance ratio.
            # ---------------------------------------------------------
            n_components = factors.shape[1] if factors is not None else 0
            explained_variance_ratio: np.ndarray | None = None
            if explained_variance is not None:
                evr_sum = float(explained_variance.sum())
                if evr_sum > 0:
                    explained_variance_ratio = explained_variance / evr_sum

            # Truncate to output_dimension if the algorithm returned more.
            if (
                self.output_dimension is not None
                and n_components > self.output_dimension
            ):
                factors = factors[:, : self.output_dimension]
                loadings = loadings[:, : self.output_dimension]
                if explained_variance is not None:
                    explained_variance = explained_variance[: self.output_dimension]
                if explained_variance_ratio is not None:
                    explained_variance_ratio = explained_variance_ratio[
                        : self.output_dimension
                    ]
                n_components = self.output_dimension

            # ---------------------------------------------------------
            # 8. Reproject.
            # ---------------------------------------------------------
            is_custom = self._is_custom_estimator()
            _mean = mean if mean is not None else 0.0
            if self.reproject in ("navigation", "both"):
                if not is_custom:
                    loadings = _reproject_navigation_scores(dc[:, _sm] - _mean, factors)
                elif hasattr(estim, "transform"):
                    loadings = estim.transform(dc[:, _sm])

            if self.reproject in ("signal", "both"):
                if not is_custom:
                    factors = _reproject_signal_components(dc[_nm, :] - _mean, loadings)
                else:
                    warnings.warn(
                        "Reprojecting the signal is not yet supported "
                        f"for algorithm='{self.algorithm}'",
                        UserWarning,
                    )
                    if self.reproject == "both":
                        self.reproject = "signal"
                    else:
                        self.reproject = None

            # ---------------------------------------------------------
            # 9. Rescale after Keenan-Kotula normalization.
            # ---------------------------------------------------------
            if self.normalize_poissonian_noise:
                _bh = self._root_bH.ravel()
                if not isinstance(_sm, slice):
                    _bh = _bh[~signal_mask]
                factors = factors * _bh[:, np.newaxis]
                _ag = self._root_aG.ravel()
                if not isinstance(_nm, slice):
                    _ag = _ag[~navigation_mask]
                loadings = loadings * _ag[:, np.newaxis]

            # ---------------------------------------------------------
            # 10. NaN-expand masked positions.
            # ---------------------------------------------------------
            if not isinstance(signal_mask, slice):
                if self.reproject not in ("both", "signal"):
                    factors = _nan_expand_rows(factors, signal_mask, dc.shape[-1])

            if not isinstance(navigation_mask, slice):
                if self.reproject not in ("both", "navigation"):
                    loadings = _nan_expand_rows(loadings, navigation_mask, dc.shape[0])

        finally:
            # ---------------------------------------------------------
            # 11. Reverse pre-treatment modifications.
            # ---------------------------------------------------------
            if mean is not None:
                if (navigation_mask is None or isinstance(navigation_mask, slice)) and (
                    signal_mask is None or isinstance(signal_mask, slice)
                ):
                    signal.data += mean

            if (
                self.normalize_poissonian_noise
                and not self.copy
                and self._root_aG is not None
            ):
                self._reverse_kk_scaling(signal)

            if self._unfolded:
                signal.fold()
                self._unfolded = False

        # Print info
        if self.print_info:
            print("\n".join(to_print_lines))

        # Build result
        result = DecompositionResult(
            components=factors,
            scores=loadings,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            mean=mean,
            bH=sqrt_bH,
            aG=sqrt_aG,
            n_components=n_components,
            centre=self.centre,
            algorithm=str(algorithm),
            params=self._build_params_dict(estim),
        )
        return result

    # ------------------------------------------------------------------
    # Algorithm dispatch
    # ------------------------------------------------------------------

    def _dispatch_algorithm(
        self, data_: np.ndarray, to_print: list[str]
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, Any]:
        """Route *data_* to the correct algorithm and return results.

        Returns
        -------
        factors : ndarray or None
            Shape ``(signal_channels, n_components)``.
        loadings : ndarray or None
            Shape ``(navigation_pixels, n_components)``.
        explained_variance : ndarray or None
        estim : sklearn estimator or None
            The fitted estimator, if applicable.
        """
        algorithm = self.algorithm

        # --- SVD path ---
        if algorithm == "SVD":
            return self._dispatch_svd(data_, to_print)

        # --- sklearn path ---
        if algorithm in _SKLEARN_ALGORITHM_NAMES:
            if not SKLEARN_INSTALLED:
                raise ImportError(f"algorithm='{algorithm}' requires scikit-learn.")
            return self._dispatch_sklearn(data_, algorithm, to_print)

        # --- Custom estimator ---
        if self._is_custom_estimator():
            return self._dispatch_custom(data_, algorithm, to_print)

        # --- Deferred paths ---
        if algorithm in _DEFERRED_ALGORITHMS:
            raise NotImplementedError(
                f"algorithm='{algorithm}' is deferred to Task 4c. "
                "Only SVD, sklearn_pca, NMF, sparse_pca, "
                "mini_batch_sparse_pca, and custom estimators are "
                "supported in this release."
            )

        raise ValueError(
            f"algorithm={algorithm!r} not recognised. Expected one of: "
            '"SVD", "MLPCA", "sklearn_pca", "NMF", "sparse_pca", '
            '"mini_batch_sparse_pca", "RPCA", "ORPCA", "ORNMF", '
            "or a custom estimator object."
        )

    def _dispatch_svd(
        self, data_: np.ndarray, to_print: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]:
        """Run SVD via :class:`hyperspy_ml_algorithms.SVDPCA`."""
        from hyperspy_ml_algorithms import SVDPCA

        # Map HyperSpy centre names to SVDPCA sklearn convention:
        # 'navigation' → 'features' (mean over axis=0/samples)
        # 'signal'     → 'signal'   (mean over axis=1/features)
        centre_arg: str | None
        if self.centre == "navigation":
            centre_arg = "features"
        elif self.centre == "signal":
            centre_arg = "signal"
        else:
            centre_arg = None

        svd = SVDPCA(
            n_components=self.output_dimension,
            svd_solver=self.svd_solver,
            centre=centre_arg,
            auto_transpose=self.auto_transpose,
            **self._extra_kwargs,
        )
        loadings = svd.fit_transform(data_)
        factors = svd.components_.T
        explained_variance = getattr(svd, "explained_variance_", None)
        return factors.astype(float), loadings.astype(float), explained_variance, svd

    def _dispatch_sklearn(
        self, data_: np.ndarray, algorithm: str, to_print: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]:
        """Run a sklearn decomposition algorithm."""
        estim = _make_sklearn_estimator(
            algorithm, self.output_dimension, **self._extra_kwargs
        )
        to_print.extend(["scikit-learn estimator:", repr(estim)])

        if hasattr(estim, "fit_transform"):
            loadings = estim.fit_transform(data_)
        elif hasattr(estim, "fit") and hasattr(estim, "transform"):
            estim.fit(data_)
            loadings = estim.transform(data_)
        else:
            raise TypeError(
                f"Estimator {type(estim).__name__} has neither "
                "fit_transform() nor fit()+transform()."
            )

        # Unwrap pipeline / GridSearchCV to access the final estimator.
        estim_ = estim
        if hasattr(estim, "steps"):
            estim_ = estim[-1]
        elif hasattr(estim, "best_estimator_"):
            estim_ = estim.best_estimator_

        if not hasattr(estim_, "components_"):
            raise AttributeError(
                f"Fitted estimator {estim_} has no attribute 'components_'"
            )
        factors = estim_.components_.T

        explained_variance: np.ndarray | None = getattr(
            estim_, "explained_variance_", None
        )
        return factors, loadings, explained_variance, estim

    def _dispatch_custom(
        self, data_: np.ndarray, algorithm: Any, to_print: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]:
        """Run a user-supplied estimator object."""
        estim = algorithm
        to_print.extend(["Custom estimator:", repr(estim)])

        if hasattr(estim, "fit_transform"):
            loadings = estim.fit_transform(data_)
        elif hasattr(estim, "fit") and hasattr(estim, "transform"):
            estim.fit(data_)
            loadings = estim.transform(data_)
        else:
            raise TypeError(
                "Custom estimator must implement fit_transform() or fit()+transform()."
            )

        estim_ = estim
        if hasattr(estim, "steps"):
            estim_ = estim[-1]
        elif hasattr(estim, "best_estimator_"):
            estim_ = estim.best_estimator_

        if not hasattr(estim_, "components_"):
            raise AttributeError(
                f"Fitted estimator {estim_} has no attribute 'components_'"
            )
        factors = estim_.components_.T
        explained_variance = getattr(estim_, "explained_variance_", None)
        return factors, loadings, explained_variance, estim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_custom_estimator(self) -> bool:
        """Return True when *algorithm* is an object, not a string."""
        import types

        algo = self.algorithm
        if isinstance(algo, str):
            return False
        if isinstance(algo, types.ModuleType):
            return False
        return hasattr(algo, "fit_transform") or (
            hasattr(algo, "fit") and hasattr(algo, "transform")
        )

    def _normalise_navigation_mask(self, mask):
        """Normalise the navigation mask, handling transposition.

        HyperSpy display order reverses navigation axes relative to
        NumPy array order.  When the mask is not a BaseSignal we
        transpose to account for this.
        """
        from hyperspy.signal import BaseSignal

        if isinstance(mask, BaseSignal):
            return _to_flat_bool(mask)
        if hasattr(mask, "T") and not isinstance(mask, BaseSignal):
            return _to_flat_bool(mask.T)
        return _to_flat_bool(mask)

    def _reverse_kk_scaling(self, signal) -> None:
        """Reverse Keenan-Kotula scaling on the signal's data in-place."""
        am = signal.axes_manager
        nav_size = am.navigation_size
        sig_size = am.signal_size

        nm_slice = self._get_safe_keep_mask(self.navigation_mask, nav_size)
        sm_slice = self._get_safe_keep_mask(self.signal_mask, sig_size)
        combined = nm_slice[:, np.newaxis] & sm_slice[np.newaxis, :]
        coeff = (
            self._root_aG.ravel()[:, np.newaxis] * self._root_bH.ravel()[np.newaxis, :]
        )
        if self._data_was_transposed:
            combined = combined.T
            coeff = coeff.T
        signal.data[combined] *= coeff[combined]

    @staticmethod
    def _get_safe_keep_mask(mask, size):
        """Return a boolean 'keep' mask from a mask that may be None."""
        if mask is None or isinstance(mask, slice):
            return np.ones(size, dtype=bool)
        return ~mask

    def _build_params_dict(self, estim: Any = None) -> dict[str, Any]:
        """Build a provenance dictionary with stage parameters."""
        params: dict[str, Any] = {
            "normalize_poissonian_noise": self.normalize_poissonian_noise,
            "algorithm": str(self.algorithm),
            "output_dimension": self.output_dimension,
            "centre": self.centre,
            "auto_transpose": self.auto_transpose,
            "reproject": self.reproject,
            "svd_solver": self.svd_solver,
            "num_chunks": self.num_chunks,
        }
        params.update(self._extra_kwargs)
        if self.return_info and estim is not None:
            params["estimator"] = estim
        return params

    # ------------------------------------------------------------------
    # Lazy (dask-backed) execution path
    # ------------------------------------------------------------------

    def _run_lazy(self) -> DecompositionResult:
        """Run decomposition on a dask-backed (lazy) signal.

        Uses dask-native operations throughout — no eager ``.compute()``
        calls except where a small result is needed (e.g. centring mean,
        explained variance).  The default SVD solver for lazy signals is
        ``"randomized"``.

        Returns
        -------
        DecompositionResult
            Result whose components/scores are dask-backed when
            ``svd_solver='full'`` is used, and numpy otherwise.
        """
        signal = self._signal
        algorithm = self.algorithm

        # Lazy path supports SVD (all solvers) and custom estimators.
        # Non-SVD named algorithms (PCA, ORPCA, ORNMF, NMF) use partial_fit
        # iteration which is NOT dask-native — for those, collect all data
        # eagerly and dispatch through the eager path.
        _effective_solver = self.svd_solver
        if algorithm == "SVD" and _effective_solver == "auto":
            # Default to randomized for lazy signals (matches original
            # LazySignal.decomposition behaviour).
            _effective_solver = "randomized"
        _lazy_svd = algorithm == "SVD" and _effective_solver in (
            "full",
            "randomized",
            "incremental",
        )
        _can_use_dask = _lazy_svd

        to_print_lines: list[str] = [
            "Decomposition info:",
            f"  normalize_poissonian_noise={self.normalize_poissonian_noise}",
            f"  algorithm={algorithm}",
            f"  output_dimension={self.output_dimension}",
            f"  centre={self.centre}",
        ]

        if not _can_use_dask:
            # For algorithms that don't have a native dask path, collect
            # all data eagerly and delegate to the eager pipeline.
            _logger.info("Collecting lazy data for non-dask-native decomposition")
            try:
                signal.data = signal.data.compute()
            except Exception:
                pass
            return self._run()

        # -------------------------------------------------------------
        # 1. Unfold multi-dimensional data into a 2-D matrix.
        # -------------------------------------------------------------
        self._original_shape = signal.data.shape
        self._unfolded = signal.unfold()
        try:
            nav_size = signal.axes_manager.navigation_size
            ndim = signal.axes_manager.navigation_dimension
            sdim = signal.axes_manager.signal_dimension

            data = signal.data  # (nav, sig) — dask

            # ---------------------------------------------------------
            # 2. Normalise masks to flat bool arrays.
            # ---------------------------------------------------------
            nm = navigation_mask = _to_flat_bool(self.navigation_mask)
            sm = signal_mask = _to_flat_bool(self.signal_mask)

            # ---------------------------------------------------------
            # 3. Apply K-K scaling (dask-aware).
            # ---------------------------------------------------------
            mean: np.ndarray | None = None
            sqrt_aG: np.ndarray | None = None
            sqrt_bH: np.ndarray | None = None

            if self.normalize_poissonian_noise:
                from hyperspy_ml.utils.preprocessing import _keenan_kotula_scale

                data, sqrt_aG, sqrt_bH = _keenan_kotula_scale(data, nm, sm, ndim, sdim)

            # ---------------------------------------------------------
            # 4. Rechunk if num_chunks is set.
            # ---------------------------------------------------------
            if self.num_chunks is not None and self.svd_solver != "randomized":
                if self.num_chunks <= 0:
                    raise ValueError(
                        f"`num_chunks` must be positive, got {self.num_chunks!r}"
                    )
                # Rechunk navigation dimension to num_chunks blocks
                data = data.rechunk({0: max(1, nav_size // self.num_chunks)})

            # ---------------------------------------------------------
            # 5. Run SVD via dask operations.
            # ---------------------------------------------------------
            if algorithm == "SVD" and _effective_solver in ("full", "randomized"):
                from hyperspy_ml.stages._lazy_decomposition import _lazy_svd_matrix

                svd_result = _lazy_svd_matrix(
                    data,
                    svd_solver=_effective_solver,
                    centre=self.centre,
                    output_dimension=self.output_dimension,
                    navigation_mask=nm,
                    signal_mask=sm,
                )
                scores = svd_result["scores"]
                components = svd_result["components"]
                explained_variance = svd_result["explained_variance"]
                mean = svd_result["mean"]
                nav_mask_1d = svd_result["nav_mask_1d"]
                sig_mask_1d = svd_result["sig_mask_1d"]

            elif algorithm == "SVD" and _effective_solver == "incremental":
                # Incremental SVD: iterate over dask chunks, call partial_fit.
                from hyperspy.external.progressbar import progressbar

                # ISVD may live in hyperspy.learn.incremental_svd (monorepo)
                # or hyperspy_ml_algorithms (standalone).
                try:
                    from hyperspy.learn.incremental_svd import ISVD
                except ImportError:
                    from hyperspy_ml_algorithms import IncrementalSVD as ISVD

                from hyperspy_ml.stages._lazy_decomposition import (
                    _iterate_dask_chunks,
                )

                obj = ISVD(n_components=self.output_dimension)
                method = partial(obj.partial_fit)

                if self.centre is not None:
                    _D_flat = data
                    if self.centre == "navigation":
                        if nm is not None:
                            mean = _D_flat[~nm, :].mean(axis=0, keepdims=True).compute()
                        else:
                            mean = _D_flat.mean(axis=0, keepdims=True).compute()
                    else:  # signal
                        mean = _D_flat.mean(axis=1, keepdims=True).compute()
                    data = data - mean

                ndim_val = signal.axes_manager.navigation_dimension
                nav_indices = [range(len(c)) for c in data.chunks[:ndim_val]]
                nblocks = int(np.prod([len(c) for c in nav_indices]))
                import math

                num_chunks = max(1, self.num_chunks or 1)
                nblocks_needed = math.ceil(nblocks / num_chunks)

                this_data = []
                for chunk in progressbar(
                    _iterate_dask_chunks(data, nav_indices),
                    total=nblocks_needed,
                    leave=True,
                    desc="Learn",
                ):
                    this_data.append(chunk)
                    if len(this_data) == num_chunks:
                        thedata = np.concatenate(this_data, axis=0)
                        method(thedata)
                        this_data = []
                if this_data:
                    thedata = np.concatenate(this_data, axis=0)
                    method(thedata)

                S = obj.singular_values_
                n_total = obj.n_samples_seen_
                explained_variance = S**2 / n_total
                components = obj.components_.T

                # Project scores over all nav chunks
                nav_indices_full = [range(len(c)) for c in data.chunks[:ndim_val]]
                H = []
                for chunk in progressbar(
                    _iterate_dask_chunks(data, nav_indices_full),
                    total=nblocks_needed,
                    leave=True,
                    desc="Project",
                ):
                    H.append(obj.transform(chunk))
                scores = np.concatenate(H, axis=0)

                sig_mask_1d = sm
                nav_mask_1d = nm

            # ---------------------------------------------------------
            # 6. Compute explained-variance ratio.
            # ---------------------------------------------------------
            n_components = scores.shape[1] if scores is not None else 0
            explained_variance_ratio: np.ndarray | None = None
            if explained_variance is not None:
                ev_sum = float(
                    explained_variance.sum().compute()
                    if hasattr(explained_variance, "compute")
                    else explained_variance.sum()
                )
                if ev_sum > 0:
                    explained_variance_ratio = (
                        explained_variance / ev_sum
                        if not hasattr(explained_variance, "compute")
                        else explained_variance / ev_sum
                    )
                    if hasattr(explained_variance_ratio, "compute"):
                        explained_variance_ratio = explained_variance_ratio.compute()

            # ---------------------------------------------------------
            # 7. Truncate to output_dimension if needed.
            # ---------------------------------------------------------
            if (
                self.output_dimension is not None
                and n_components > self.output_dimension
            ):
                factors_idx = slice(None), slice(None, self.output_dimension)
                scores = scores[factors_idx]
                components = components[factors_idx]
                if explained_variance is not None:
                    explained_variance = explained_variance[: self.output_dimension]
                if explained_variance_ratio is not None:
                    explained_variance_ratio = explained_variance_ratio[
                        : self.output_dimension
                    ]
                n_components = self.output_dimension

            # ---------------------------------------------------------
            # 8. Mask conversion: None → slice(None).
            # ---------------------------------------------------------
            if navigation_mask is None:
                navigation_mask = slice(None)
            if signal_mask is None:
                signal_mask = slice(None)

            _nm_slice = (
                navigation_mask
                if isinstance(navigation_mask, slice)
                else ~navigation_mask
            )
            _sm_slice = signal_mask if isinstance(signal_mask, slice) else ~signal_mask

            # ---------------------------------------------------------
            # 9. Pre-compute flat masks for NaN-expand.
            # ---------------------------------------------------------
            _flat_nav_mask = _to_flat_bool(
                None if isinstance(navigation_mask, slice) else navigation_mask
            )
            _flat_sig_mask = _to_flat_bool(
                None if isinstance(signal_mask, slice) else signal_mask
            )

            # ---------------------------------------------------------
            # 10. Reproject navigation.
            # ---------------------------------------------------------
            from hyperspy_ml.stages._lazy_decomposition import (
                _lazy_reproject_navigation,
            )

            scores, _nav_reprojected = _lazy_reproject_navigation(
                self.reproject,
                algorithm,
                _effective_solver,
                scores,
                components,
                mean,
                self.centre,
                data,
                sig_mask_1d,
            )

            # ---------------------------------------------------------
            # 11. Reproject signal.
            # ---------------------------------------------------------
            _signal_reprojected = False
            if self.reproject in ("signal", "both"):
                from hyperspy_ml.stages._lazy_decomposition import (
                    _lazy_reproject_signal,
                )

                components = _lazy_reproject_signal(
                    algorithm,
                    _effective_solver,
                    self.reproject,
                    scores,
                    components,
                    mean,
                    self.centre,
                    data,
                    nav_mask_1d,
                    _flat_sig_mask,
                    _flat_nav_mask,
                )
                _signal_reprojected = True

            # ---------------------------------------------------------
            # 12. Rescale after Keenan-Kotula normalization.
            # ---------------------------------------------------------
            if self.normalize_poissonian_noise and sqrt_bH is not None:
                _bh = sqrt_bH.ravel()
                if not isinstance(signal_mask, slice) and _flat_sig_mask is not None:
                    _bh = _bh[~_flat_sig_mask]
                components = components * _bh[:, np.newaxis]

                _ag = sqrt_aG.ravel()
                if (
                    not isinstance(navigation_mask, slice)
                    and _flat_nav_mask is not None
                ):
                    _ag = _ag[~_flat_nav_mask]
                scores = scores * _ag[:, np.newaxis]

            # ---------------------------------------------------------
            # 13. NaN-expand masked positions.
            # ---------------------------------------------------------
            if not isinstance(signal_mask, slice):
                if self.reproject not in ("both", "signal"):
                    from hyperspy_ml.utils.preprocessing import _nan_expand_rows

                    if components.shape[0] != data.shape[-1]:
                        components = _nan_expand_rows(
                            components, signal_mask, data.shape[-1]
                        )
                    else:
                        # Already full size — NaN-fill masked positions.
                        components[signal_mask, :] = np.nan

            if not isinstance(navigation_mask, slice):
                if self.reproject not in ("both", "navigation"):
                    from hyperspy_ml.utils.preprocessing import _nan_expand_rows

                    if scores.shape[0] != data.shape[0]:
                        scores = _nan_expand_rows(
                            scores, navigation_mask, data.shape[0]
                        )
                    else:
                        # Already full size — NaN-fill masked positions.
                        scores[navigation_mask, :] = np.nan

        finally:
            if self._unfolded:
                signal.fold()
                self._unfolded = False

        # Print info
        if self.print_info:
            print("\n".join(to_print_lines))

        # Build result — components/scores remain dask-backed for
        # svd_solver='full', computed numpy for randomized/incremental.
        result = DecompositionResult(
            components=components,
            scores=scores,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            mean=mean,
            bH=sqrt_bH,
            aG=sqrt_aG,
            n_components=n_components,
            centre=self.centre,
            algorithm=str(algorithm),
            params=self._build_params_dict(),
        )
        return result

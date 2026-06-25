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

"""Blind source separation (BSS) stage.

Ported from :meth:`MVA.blind_source_separation` (~430 lines in
``_mva.py``).  Operates on a :class:`~hyperspy_ml.results.base.DecompositionResult`
produced by the :class:`~hyperspy_ml.stages.decomposition.Decomposition` stage.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from typing import Any

import numpy as np
from hyperspy.exceptions import VisibleDeprecationWarning

from hyperspy_ml.results.base import BSSResult
from hyperspy_ml.utils.preprocessing import (
    _get_derivative,
    _get_sklearn_algorithms,
    _normalize_components,
)

_logger = logging.getLogger(__name__)

MDP_INSTALLED = importlib.util.find_spec("mdp") is not None
SKLEARN_INSTALLED = importlib.util.find_spec("sklearn") is not None

_MDP_ALGORITHMS = frozenset({"FastICA", "JADE", "CuBICA", "TDSEP"})


def _orthomax(A, gamma=1.0, reltol=1.4901e-07, maxit=256):
    """Orthogonal rotation of FA or PCA loadings (varimax when gamma=1).

    Ported from the original ``hyperspy.misc.utils.orthomax``
    (commit ee35e4411).  Returns ``(B, T)`` where *B* is the rotated
    loadings matrix and *T* the orthogonal rotation matrix.
    """
    d, m = A.shape
    B = A.copy()
    T = np.eye(m)

    if 0 <= gamma <= 1:
        # Fast Lawley & Maxwell iteration
        converged = False
        while not converged:
            D = 0.0
            for _k in range(1, maxit + 1):
                Dold = D
                tmp11 = np.sum(B**2, axis=0)
                tmp1 = np.diag(np.array(tmp11).flatten())
                tmp2 = gamma * B
                tmp3 = d * B**3
                L, Dvals, M = np.linalg.svd(A.T @ (tmp3 - tmp2 @ tmp1))
                T = L @ M
                D = np.sum(Dvals)
                B = A @ T
                if np.abs(D - Dold) / D < reltol:
                    converged = True
                    break
    else:
        # Sequence of bivariate rotations
        for _iter in range(1, maxit + 1):
            maxTheta = 0.0
            for i in range(0, m - 1):
                for j in range(i + 1, m):
                    Bi = B[:, i]
                    Bj = B[:, j]
                    u = Bi**2 - Bj**2
                    v = 2 * Bi * Bj
                    usum = u.sum()
                    vsum = v.sum()
                    numer = 2 * u @ v - 2 * gamma * usum * vsum / d
                    denom = u @ u - v @ v - gamma * (usum**2 - vsum**2) / d
                    theta = np.arctan2(numer, denom) / 4
                    maxTheta = max(maxTheta, np.abs(theta))
                    Tij = np.array(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)],
                        ]
                    )
                    B[:, [i, j]] = B[:, [i, j]] @ Tij
                    T[:, [i, j]] = T[:, [i, j]] @ Tij
            if maxTheta < reltol:
                break
    return B, T


class BSS:
    """Standalone blind source separation stage.

    Parameters
    ----------
    number_of_components : int or None
    algorithm : str or object, default ``"orthomax"``
        ``"sklearn_fastica"``, ``"orthomax"``, MDP names (`FastICA`,
        `JADE`, `CuBICA`, `TDSEP`), or a custom object.
    diff_order : int, default 0
        Derivative order for preprocessing.
    diff_axes : list or None
        Axes to differentiate.
    comp_list : list or None
        Specific component indices.
    mask : BaseSignal or None
        Navigation mask.
    on_scores : bool, default False
        Apply BSS on scores (True) or components (False).
    reverse_component_criterion : str
        Criterion for auto-reversing component sign.
    whiten_method : str or None, default ``"PCA"``
        Whitening method (``"PCA"``, ``"ZCA"``, or ``None``).
    return_info : bool, default False
    print_info : bool, default True
    **kwargs
        Passed to the BSS algorithm.
    """

    def __init__(
        self,
        number_of_components: int | None = None,
        algorithm: str | object = "orthomax",
        diff_order: int = 0,
        diff_axes: Any = None,
        comp_list: list[int] | None = None,
        mask: Any = None,
        on_scores: bool = False,
        reverse_component_criterion: str = "components",
        whiten_method: str | None = "PCA",
        return_info: bool = False,
        print_info: bool = True,
        **kwargs: Any,
    ) -> None:
        self.number_of_components = number_of_components
        self.algorithm = algorithm
        self.diff_order = diff_order
        self.diff_axes = diff_axes
        self.comp_list = comp_list
        self.mask = mask
        self.on_scores = on_scores
        self.reverse_component_criterion = reverse_component_criterion
        self.whiten_method = whiten_method
        self.return_info = return_info
        self.print_info = print_info
        self._extra_kwargs = kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, decomposition_result) -> tuple[BSSResult, Any]:
        """Run BSS on a decomposition result.

        Parameters
        ----------
        decomposition_result : DecompositionResult
            Output from :class:`Decomposition`.

        Returns
        -------
        bss_result : BSSResult
        return_info : object or None
        """
        return self._run(decomposition_result)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run(self, lr):
        """Execute the full BSS pipeline."""
        from hyperspy.signal import BaseSignal

        factors = None
        if not hasattr(lr, "components") or lr.components is None:
            raise AttributeError(
                "A decomposition must be performed before blind "
                "source separation, or factors must be provided."
            )
        if self.on_scores:
            factors = _make_signal_from(lr.scores, lr)
        else:
            factors = _make_signal_from(lr.components, lr)

        if hasattr(factors, "compute"):
            factors.compute()

        if not isinstance(factors, BaseSignal):
            raise TypeError(
                f"`factors` must be a BaseSignal instance, got {type(factors)}"
            )

        if factors.axes_manager.navigation_dimension != 1:
            raise ValueError("`factors` must have navigation dimension == 1")
        if factors.axes_manager.navigation_size < 2:
            raise ValueError("`factors` must have navigation size > 1")

        mask = self.mask
        if mask is not None:
            if not isinstance(mask, BaseSignal):
                raise ValueError("`mask` must be a HyperSpy signal.")
            mask = mask.deepcopy()
            if hasattr(mask, "compute"):
                mask.compute()

        # Component selection
        kw = self._extra_kwargs
        number_of_components = self.number_of_components
        comp_list = self.comp_list

        if number_of_components is not None:
            comp_list = list(range(number_of_components))
        elif comp_list is not None:
            number_of_components = len(comp_list)
        else:
            if lr.n_components > 0:
                number_of_components = lr.n_components
                comp_list = list(range(number_of_components))
            else:
                raise ValueError("No `number_of_components` or `comp_list` provided.")

        from hyperspy.misc import utils as _utils

        factors = _utils.stack([factors.inav[i] for i in comp_list])

        # Derivative preprocessing
        diff_axes = self.diff_axes
        diff_order = self.diff_order
        if diff_order > 0:
            factors = _get_derivative(
                factors, diff_axes=diff_axes, diff_order=diff_order
            )
            if mask is not None:
                mask_diff_axes = (
                    [iaxis - 1 for iaxis in diff_axes]
                    if diff_axes is not None
                    else None
                )
                mask.change_dtype("float")
                mask.data[mask.data == 1] = np.nan
                mask = _get_derivative(
                    mask, diff_axes=mask_diff_axes, diff_order=diff_order
                )
                mask.data[np.isnan(mask.data)] = 1
                mask.change_dtype("bool")

        factors.unfold()
        if mask is not None:
            mask.unfold()
            factors_data = factors.data.T[np.where(~mask.data)]
        else:
            factors_data = factors.data.T

        # Whitening
        whiten_method = self.whiten_method
        if whiten_method is not None:
            _logger.info(f"Whitening with method '{whiten_method}'")
            from hyperspy_ml_algorithms import Whitening

            whitener = Whitening(method=whiten_method)
            factors_data = whitener.fit_transform(factors_data)
            invsqcovmat = whitener.whitening_matrix_
        else:
            invsqcovmat = None

        # Dispatch algorithm
        algorithm = self.algorithm
        to_print: list[str] = [
            "Blind source separation info:",
            f"  number_of_components={number_of_components}",
            f"  algorithm={algorithm}",
            f"  diff_order={diff_order}",
            f"  reverse_component_criterion={self.reverse_component_criterion}",
            f"  whiten_method={whiten_method}",
        ]
        to_return_info = None

        if algorithm == "orthomax":
            _, unmixing_matrix = _orthomax(factors_data, **kw)

        elif algorithm == "sklearn_fastica":
            if not SKLEARN_INSTALLED:
                raise ImportError(f"algorithm='{algorithm}' requires scikit-learn")
            if not kw.get("tol", False):
                kw["tol"] = 1e-10
            estim = _get_sklearn_algorithms(algorithm)(**kw)
            if estim.whiten and whiten_method is not None:
                _logger.warning(
                    "HyperSpy performs its own whitening, "
                    f"sklearn whiten is disabled for '{algorithm}'"
                )
                estim.whiten = False
            _ = estim.fit_transform(factors_data)
            unmixing_matrix = estim.components_
            to_print.extend(["scikit-learn estimator:", str(estim)])
            if self.return_info:
                to_return_info = estim

        elif algorithm in _MDP_ALGORITHMS:
            if not MDP_INSTALLED:
                raise ImportError(
                    f"algorithm='{algorithm}' requires MDP toolbox. "
                    "Install with: pip install mdp"
                )
            import mdp  # noqa: F811

            temp_func = getattr(mdp.nodes, algorithm + "Node")
            node = temp_func(**kw)
            node.train(factors_data)
            unmixing_matrix = node.get_recmatrix()
            to_print.extend(["mdp estimator:", str(node)])
            if self.return_info:
                to_return_info = node

        elif hasattr(algorithm, "fit_transform") or (
            hasattr(algorithm, "fit") and hasattr(algorithm, "transform")
        ):
            estim = algorithm
            if hasattr(estim, "fit_transform"):
                _ = estim.fit_transform(factors_data)
            else:
                estim.fit(factors_data)
            if hasattr(estim, "steps"):
                estim_ = estim[-1]
            elif hasattr(estim, "best_estimator_"):
                estim_ = estim.best_estimator_
            else:
                estim_ = estim
            if hasattr(estim_, "components_"):
                unmixing_matrix = estim_.components_
            else:
                raise AttributeError(f"Fitted estimator {estim_} has no 'components_'")
            to_print.extend(["Custom estimator:", str(estim)])
            if self.return_info:
                to_return_info = estim
        else:
            raise ValueError(f"'algorithm' not recognised: {algorithm!r}")

        # Apply whitening to unmixing matrix
        if invsqcovmat is not None:
            w = unmixing_matrix @ invsqcovmat
        else:
            w = unmixing_matrix

        # Sort components by explained-variance-weighted unmixing
        if lr.explained_variance is not None:
            if hasattr(lr.explained_variance, "compute"):
                lr.explained_variance = lr.explained_variance.compute()
            sorting = np.argsort(
                lr.explained_variance[:number_of_components] @ abs(w.T)
            )[::-1]
            w = w[sorting, :]

        # Unmix components
        bss_components, bss_scores = _unmix_components(lr, w, self.on_scores)
        _auto_reverse_bss_component(
            bss_components, bss_scores, self.reverse_component_criterion
        )

        if self.print_info:
            print("\n".join(to_print))

        result = BSSResult(
            bss_components=bss_components,
            bss_scores=bss_scores,
            unmixing_matrix=w,
            bss_algorithm=str(algorithm),
            on_scores=self.on_scores,
            params=self._build_params(),
        )
        return result, to_return_info

    def _build_params(self) -> dict[str, Any]:
        return {
            "number_of_components": self.number_of_components,
            "algorithm": str(self.algorithm),
            "diff_order": self.diff_order,
            "on_scores": self.on_scores,
            "whiten_method": self.whiten_method,
            **self._extra_kwargs,
        }

    # ------------------------------------------------------------------
    # Component manipulation (operate on a BSSResult)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_bss_components(result, target="components", function=np.sum):
        """Normalize BSS components."""
        if target in ("factors", "loadings"):
            alt = "components" if target == "factors" else "scores"
            warnings.warn(
                f'`target="{target}"` is deprecated, use `target="{alt}"` instead.',
                VisibleDeprecationWarning,
            )
            if target == "factors":
                target_arr, other = result.bss_components, result.bss_scores
            else:
                target_arr, other = result.bss_scores, result.bss_components
        elif target == "components":
            target_arr, other = result.bss_components, result.bss_scores
        elif target == "scores":
            target_arr, other = result.bss_scores, result.bss_components
        else:
            raise ValueError(
                'target must be "components", "scores", "factors", or "loadings"'
            )
        if target_arr is None:
            raise ValueError("BSS must be performed before normalization")
        _normalize_components(target=target_arr, other=other, function=function)

    @staticmethod
    def reverse_bss_component(result, component_number):
        """Reverse a BSS component."""
        if hasattr(result.bss_components, "compute"):
            _logger.warning(
                f"Component(s) {component_number} not reversed, "
                "feature not implemented for lazy computations"
            )
            return
        for i in [component_number]:
            result.bss_components[:, i] *= -1
            result.bss_scores[:, i] *= -1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_signal_from(array, source_result):
    """Wrap a numpy array as a Signal1D for use as BSS factors."""
    from hyperspy.signals import Signal1D

    return Signal1D(np.asarray(array, dtype=float))


def _unmix_components(lr, w, on_scores):
    """Apply the unmixing matrix to produce BSS components / scores."""
    n = w.shape[1]
    try:
        w_inv = np.linalg.inv(w)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Singular unmixing matrix — using pinv instead.",
            UserWarning,
        )
        w_inv = np.linalg.pinv(w)

    if on_scores:
        bss_scores = lr.scores[:, :n] @ w.T
        bss_components = lr.components[:, :n] @ w_inv
    else:
        bss_components = lr.components[:, :n] @ w.T
        bss_scores = lr.scores[:, :n] @ w_inv
    return bss_components, bss_scores


def _auto_reverse_bss_component(bss_components, bss_scores, criterion):
    """Auto-reverse components based on asymmetry criterion."""
    values = bss_components if criterion in ("components", "factors") else bss_scores
    n = bss_components.shape[1]
    for i in range(n):
        v = values[:, i]
        mn, mx = np.nanmin(v), np.nanmax(v)
        if mn < 0 and -mn > mx:
            bss_components[:, i] *= -1
            bss_scores[:, i] *= -1

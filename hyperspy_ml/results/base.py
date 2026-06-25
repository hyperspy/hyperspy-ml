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

"""Typed result containers for ML pipeline stages with model reconstruction.

DecompositionResult now includes model reconstruction
(:meth:`~DecompositionResult.get_decomposition_model` and
:meth:`~DecompositionResult.get_bss_model`) with full dask/einsum lazy
support, plus crop, transpose, rescaling, and restore utilities.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from hyperspy.exceptions import VisibleDeprecationWarning


@dataclass
class DecompositionResult:
    """Container for the output of a matrix decomposition.

    Attributes
    ----------
    components : ndarray, shape (signal_channels, n_components)
        Factor matrix (transposed components_ from sklearn convention).
        Rows are signal channels, columns are components.
    scores : ndarray, shape (navigation_pixels, n_components)
        Loading matrix.  Rows are navigation positions, columns are components.
    bss_components : ndarray or None
        BSS factor matrix, populated after blind source separation.
    bss_scores : ndarray or None
        BSS loading matrix, populated after blind source separation.
    explained_variance : ndarray or None, shape (n_components,)
        Raw explained variance per component (eigenvalues of the scatter
        matrix).  ``None`` when the algorithm does not provide it.
    explained_variance_ratio : ndarray or None, shape (n_components,)
        Fraction of total explained variance per component
        (``explained_variance / sum``).  Computed automatically when
        ``explained_variance`` is available.
    mean : ndarray or None
        The mean that was subtracted during centering.  ``None`` when
        ``centre`` is ``None``.
    bH : ndarray or None
        Square-root of signal-channel Poisson-noise weighting
        (:math:`\\sqrt{bH}`), computed during Keenan-Kotula scaling.
        ``None`` when ``normalize_poissonian_noise`` is ``False``.
    aG : ndarray or None
        Square-root of navigation-pixel Poisson-noise weighting
        (:math:`\\sqrt{aG}`), computed during Keenan-Kotula scaling.
        ``None`` when ``normalize_poissonian_noise`` is ``False``.
    n_components : int
        Number of components actually extracted.
    centre : str or None
        Centering mode that was applied (``"navigation"``, ``"signal"``, or
        ``None``).
    algorithm : str or None
        Name of the decomposition algorithm.
    params : dict
        All parameters that were passed to the stage constructor (for
        provenance and reproducibility).
    _nav_shape : tuple or None
        Multi-dimensional navigation shape in NumPy (fastest-varying-first)
        order, e.g. ``(ny, nx)``.  Used by model reconstruction to reshape
        scores into the original data geometry.  ``None`` if no source
        signal geometry was recorded.
    _source_signal : BaseSignal or None
        Reference to the original signal used for deepcopy during model
        reconstruction.  ``None`` when model reconstruction via
        ``source_signal=`` argument.
    """

    components: np.ndarray | None = None
    scores: np.ndarray | None = None
    bss_components: np.ndarray | None = None
    bss_scores: np.ndarray | None = None
    explained_variance: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None
    mean: np.ndarray | None = None
    bH: np.ndarray | None = None
    aG: np.ndarray | None = None
    n_components: int = 0
    centre: str | None = None
    algorithm: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    _nav_shape: tuple | None = field(default=None, repr=False)
    _source_signal: Any = field(default=None, repr=False)
    _data_before_treatments: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialise the events namespace."""
        object.__setattr__(self, "events", _ResultEvents())

    def _notify_data_changed(self) -> None:
        """Fire :attr:`events.data_changed`."""
        self.events.data_changed = True

    # ------------------------------------------------------------------
    # Model reconstruction
    # ------------------------------------------------------------------

    def get_decomposition_model(
        self,
        source_signal=None,
        components=None,
        chunks="auto",
        lazy_output=None,
    ):
        """Rebuild a signal from the decomposition components and scores.

        Parameters
        ----------
        source_signal : BaseSignal or None
            Original signal used as a template (metadata, axes, shape).
            If *self* was produced with ``_source_signal`` set, that signal
            is used as the default when *source_signal* is ``None``.
        components : None, int, or list of int, default None
            * ``None`` — rebuild from all components.
            * ``int``  — rebuild from components ``0`` … ``<int>``.
            * ``list[int]`` — rebuild from only the listed component indices.
        chunks : str, int, or tuple, default ``"auto"``
            Controls chunking when ``lazy_output=True`` (see Notes).
        lazy_output : bool or None, default None
            * ``True`` — return a :class:`~hyperspy.api.signals.LazySignal`
              that defers computation (never calls ``.compute()``).
            * ``False`` — return an eager :class:`~.api.signals.BaseSignal`.
            * ``None`` — use ``source_signal._lazy`` to decide.

        Returns
        -------
        :class:`~hyperspy.api.signals.BaseSignal` or subclass
            The reconstructed model signal.

        Notes
        -----
        The ``chunks`` parameter can be:

        * ``"auto"`` — dask chooses chunk sizes.
        * ``int`` — split the signal dimension into chunks of this size;
          navigation uses ``"auto"``.
        * ``tuple(sig_chunks, nav_chunks)`` — explicit control.
        """
        return self._calculate_recmatrix(
            source_signal=source_signal,
            components=components,
            mva_type="decomposition",
            chunks=chunks,
            lazy_output=lazy_output,
        )

    def get_bss_model(
        self,
        source_signal=None,
        components=None,
        chunks="auto",
        lazy_output=None,
    ):
        """Rebuild a signal from the BSS components and scores.

        See :meth:`get_decomposition_model` for parameter details.
        """
        return self._calculate_recmatrix(
            source_signal=source_signal,
            components=components,
            mva_type="bss",
            chunks=chunks,
            lazy_output=lazy_output,
        )

    def _calculate_recmatrix(
        self,
        source_signal=None,
        components=None,
        mva_type="decomposition",
        chunks="auto",
        lazy_output=None,
    ):
        """Core model-reconstruction logic — ported from
        :meth:`hyperspy.learn._mva.MVA._calculate_recmatrix`.

        Parameters
        ----------
        source_signal : BaseSignal or None
        components : None, int, or list[int]
        mva_type : str {"decomposition", "bss"}
        chunks : str, int, or tuple
        lazy_output : bool or None

        Returns
        -------
        BaseSignal or LazySignal
        """
        sig = source_signal if source_signal is not None else self._source_signal
        if sig is None:
            raise ValueError(
                "No source_signal provided and _source_signal is None. "
                "Pass source_signal= to get_decomposition_model() or "
                "store it on the result."
            )

        if mva_type == "decomposition":
            factors = self.components
            loadings = self.scores
        elif mva_type == "bss":
            factors = self.bss_components
            loadings = self.bss_scores
        else:
            raise ValueError(
                f"mva_type must be 'decomposition' or 'bss', not {mva_type!r}"
            )

        if factors is None or loadings is None:
            raise ValueError(
                f"No {mva_type} results available. Run decomposition or BSS first."
            )

        if components is None:
            signal_name = f"model from {mva_type} with {factors.shape[1]} components"
        elif hasattr(components, "__iter__"):
            components = list(components)
            factors = factors[:, components]
            loadings = loadings[:, components]
            signal_name = f"model from {mva_type} with components {components}"
        else:
            factors = factors[:, :components]
            loadings = loadings[:, :components]
            signal_name = f"model from {mva_type} with {components} components"

        # Determine nav_shape: prefer explicit, then stored, then infer.
        nav_shape = self._nav_shape
        if nav_shape is None:
            nav_shape = sig.axes_manager.navigation_shape[::-1]

        if lazy_output is None:
            lazy_output = getattr(sig, "_lazy", False)

        # Parse chunks parameter into nav / sig chunk specs.
        if isinstance(chunks, (tuple, list)):
            sig_chunks = chunks[0] if len(chunks) >= 1 else "auto"
            nav_chunks = chunks[1] if len(chunks) >= 2 else "auto"
        else:
            sig_chunks = chunks
            nav_chunks = "auto"

        n_comp = loadings.shape[1]

        if lazy_output or (hasattr(sig, "_lazy") and sig._lazy):
            import dask.array as da

            if isinstance(loadings, da.Array):
                loadings_np = loadings.compute()
            else:
                loadings_np = loadings
            loadings_3d = loadings_np.reshape(nav_shape + (n_comp,))

            if isinstance(factors, da.Array):
                factors_da = factors.rechunk((sig_chunks, -1))
            else:
                factors_da = da.from_array(factors, chunks=(sig_chunks, -1))
            loadings_da = da.from_array(
                loadings_3d,
                chunks=(nav_chunks,) * len(nav_shape) + (-1,),
            )

            a = da.einsum("sc,...c->...s", factors_da, loadings_da)
        else:
            loadings_t = loadings.T
            a = factors @ loadings_t
            a = a.T.reshape(nav_shape + (factors.shape[0],))

        sc = sig.deepcopy()
        sc.data = a
        sc.metadata.General.title += " " + signal_name

        if self.mean is not None:
            mean_arr = np.asarray(self.mean)
            if self.centre == "navigation":
                sc.data += mean_arr.reshape((1,) * len(nav_shape) + (factors.shape[0],))
            elif self.centre == "signal":
                sc.data += mean_arr.reshape(nav_shape + (1,))
            else:
                sc.data += mean_arr.reshape(nav_shape + (-1,))

        if lazy_output:
            if not getattr(sc, "_lazy", False):
                sc = sc.as_lazy()
            sc.rechunk(nav_chunks=nav_chunks, sig_chunks=sig_chunks, inplace=True)
        elif getattr(sc, "_lazy", False):
            sc.compute()

        return sc

    # ------------------------------------------------------------------
    # Dimension cropping
    # ------------------------------------------------------------------

    def crop_decomposition_dimension(self, n, compute=False):
        """Crop the decomposition results to *n* components.

        Parameters
        ----------
        n : int
            Number of components to keep.
        compute : bool, default False
            If ``True`` and results are dask-backed, materialise them.
        """
        self.components = self.components[:, :n]
        self.scores = self.scores[:, :n]
        if self.explained_variance is not None:
            self.explained_variance = self.explained_variance[:n]
        if self.explained_variance_ratio is not None:
            self.explained_variance_ratio = self.explained_variance_ratio[:n]
        self.n_components = n

        if compute:
            try:
                import dask.array as da
            except ImportError:
                return
            if isinstance(self.components, da.Array):
                self.components = self.components.compute()
            if isinstance(self.scores, da.Array):
                self.scores = self.scores.compute()
            if isinstance(self.explained_variance, da.Array):
                self.explained_variance = self.explained_variance.compute()
            if isinstance(self.explained_variance_ratio, da.Array):
                self.explained_variance_ratio = self.explained_variance_ratio.compute()

        self._notify_data_changed()

    # ------------------------------------------------------------------
    # Transpose
    # ------------------------------------------------------------------

    def _transpose_results(self):
        """Swap components ↔ scores and bss_components ↔ bss_scores."""
        (
            self.components,
            self.scores,
            self.bss_components,
            self.bss_scores,
        ) = (
            self.scores,
            self.components,
            self.bss_scores,
            self.bss_components,
        )

    # ------------------------------------------------------------------
    # Explained-variance ratio computation
    # ------------------------------------------------------------------

    def _compute_explained_variance_ratio(self):
        """Compute :attr:`explained_variance_ratio` from
        :attr:`explained_variance`.

        If :attr:`explained_variance` is a dask array it is computed
        eagerly (the array is small — ``n_components`` elements).
        """
        ev = self.explained_variance
        if ev is None:
            self.explained_variance_ratio = None
            return

        try:
            import dask.array as da

            if isinstance(ev, da.Array):
                ev = ev.compute()
        except ImportError:
            pass

        evr_sum = float(np.sum(ev))
        if evr_sum > 0:
            self.explained_variance_ratio = ev / evr_sum
        else:
            self.explained_variance_ratio = None

    # ------------------------------------------------------------------
    # Pre-treatment helpers (deprecated path compatibility)
    # ------------------------------------------------------------------

    def normalize_poissonian_noise(
        self, source_signal=None, navigation_mask=None, signal_mask=None
    ):
        """Apply Keenan-Kotula Poisson-noise scaling to *source_signal*.

        .. deprecated:: 2.5
           Pre-treatment is now reversed mathematically after
           decomposition; explicit scaling is no longer needed.

        Parameters
        ----------
        source_signal : BaseSignal or None
        navigation_mask : ndarray or None
        signal_mask : ndarray or None
        """
        from hyperspy_ml.utils.preprocessing import _keenan_kotula_scale

        sig = source_signal or self._source_signal
        if sig is None:
            raise ValueError("No signal available for scaling")

        sig.unfold()
        try:
            data_2d = sig.data
            if sig.axes_manager[0].index_in_array != 0:
                data_2d = data_2d.T

            from hyperspy_ml.utils.preprocessing import _to_flat_bool

            nav_mask = _to_flat_bool(navigation_mask)
            sig_mask = _to_flat_bool(signal_mask)
            scaled, root_aG, root_bH = _keenan_kotula_scale(
                data_2d, nav_mask, sig_mask, ndim=1, sdim=1
            )
            if sig.axes_manager[0].index_in_array != 0:
                scaled = scaled.T

            self._data_before_treatments = (
                sig.data.copy() if not getattr(sig, "_lazy", False) else sig.data
            )
            sig.data = scaled
            self.bH = root_bH
            self.aG = root_aG
        finally:
            sig.fold()

    def undo_treatments(self, source_signal=None):
        """Restore data to the state before pre-treatments.

        .. deprecated:: 2.5
           Pre-treatment data modifications are now reversed
           mathematically after decomposition.
        """
        warnings.warn(
            "undo_treatments() is deprecated and will be removed in a "
            "future release.  Data modifications are now reversed "
            "mathematically after decomposition.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )

        sig = source_signal or self._source_signal
        if self._data_before_treatments is not None and sig is not None:
            folded = False
            if sig.axes_manager.navigation_size != sig.data.shape[0]:
                sig.unfold()
                folded = True
            sig.data[:] = self._data_before_treatments
            if folded:
                sig.fold()
            self._data_before_treatments = None
        else:
            raise AttributeError(
                "Unable to undo data pre-treatments! Ensure "
                "normalize_poissonian_noise() was called first or set "
                "copy=True when calling the decomposition."
            )


class _ResultEvents:
    """Minimal events namespace for result containers."""

    data_changed: bool = False

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

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from hyperspy.exceptions import VisibleDeprecationWarning

_logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Scree plot
    # ------------------------------------------------------------------

    def get_scree_plot_data(self):
        """Return scree plot data as a Signal1D.

        Returns the explained variance ratio (centred decomposition) or
        proportion of total variation (uncentred) as a function of
        component index.

        Returns
        -------
        s : Signal1D
            Variance data for scree plotting.

        Raises
        ------
        AttributeError
            If ``explained_variance_ratio`` is ``None``.
        """
        from hyperspy.signals import Signal1D

        if self.explained_variance_ratio is None:
            raise AttributeError(
                "The explained_variance_ratio attribute is "
                "`None`, did you forget to run decomposition()?"
            )
        is_centred = self.centre is not None
        algorithm = self.algorithm or "Decomposition"
        scree_title = f"{algorithm} Scree Plot"
        component_label = (
            "Principal component index" if is_centred else "Component index"
        )
        s = Signal1D(self.explained_variance_ratio)
        s.metadata.General.title = (
            getattr(self, "_source_title", "") + "\n" + scree_title
        )
        s.axes_manager[-1].name = component_label
        s.axes_manager[-1].units = ""
        return s

    def get_explained_variance_ratio(self, *args, **kwargs):
        """Deprecated: use :meth:`get_scree_plot_data` instead."""
        warnings.warn(
            "get_explained_variance_ratio() is deprecated, "
            "use get_scree_plot_data() instead.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        return self.get_scree_plot_data(*args, **kwargs)

    def plot_scree(
        self,
        n=30,
        log=True,
        threshold=0,
        hline="auto",
        vline=False,
        xaxis_type="index",
        xaxis_labeling=None,
        signal_fmt=None,
        noise_fmt=None,
        fig=None,
        ax=None,
        **kwargs,
    ):
        """Plot the decomposition scree plot.

        For centred decompositions (e.g. PCA), the y-axis shows
        explained variance ratio. For uncentred decompositions, it
        shows the proportion of total variation.

        Parameters
        ----------
        n : int or None, default 30
            Number of components to plot. ``None`` shows all.
        log : bool, default True
            If True, use log scale for y-axis.
        threshold : float or int, default 0
            For float: cutoff proportion for signal/noise.
            For int: number of signal components.
        hline : {'auto', True, False}, default 'auto'
            Draw a horizontal line at the cutoff value.
        vline : bool, default False
            Draw a vertical line at the elbow estimate.
        xaxis_type : {'index', 'number'}, default 'index'
            X-axis labeling: 0-based or 1-based.
        xaxis_labeling : {'ordinal', 'cardinal', None}
            Label format. ``None`` auto-selects.
        signal_fmt : dict or None
            Matplotlib formatting for signal components.
        noise_fmt : dict or None
            Matplotlib formatting for noise components.
        fig : matplotlib.figure.Figure or None
            Figure to draw into.
        ax : matplotlib.axes.Axes or None
            Axes to draw into.
        **kwargs
            Passed to ``plt.figure(**kwargs)``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from hyperspy.misc import utils as hs_utils
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        s = self.get_scree_plot_data()
        if hs_utils.is_cupy_array(s.data):
            s.to_host()

        n_max = len(self.explained_variance_ratio)
        if n is None:
            n = n_max
        elif n > n_max:
            n = n_max

        # Determine right number of components for signal and cutoff value
        if isinstance(threshold, float):
            if not 0 < threshold < 1:
                raise ValueError("Variance threshold should be between 0 and 1")
            if threshold < s.data.min():
                n_signal_pcs = n
            else:
                n_signal_pcs = np.where((s < threshold).data)[0][0]
        else:
            n_signal_pcs = threshold
            if n_signal_pcs == 0:
                hline = False

        if vline:
            nsc = self._get_number_significant_components()
            if nsc is None:
                vline = False
            else:
                index_number_significant_components = nsc - 1
        else:
            vline = False

        # Handling hline logic
        if hline == "auto":
            if isinstance(threshold, float):
                cutoff = threshold
            else:
                hline = False
        elif hline:
            if isinstance(threshold, float):
                cutoff = threshold
            elif n_signal_pcs > 0:
                cutoff = s.data[n_signal_pcs - 1]
        else:
            hline = False

        # Default formatting
        if signal_fmt is None:
            signal_fmt = {
                "c": "#C24D52",
                "linestyle": "",
                "marker": "^",
                "markersize": 10,
                "zorder": 3,
            }
        if noise_fmt is None:
            noise_fmt = {
                "c": "#4A70B0",
                "linestyle": "",
                "marker": "o",
                "markersize": 10,
                "zorder": 3,
            }

        if xaxis_labeling is None:
            xaxis_labeling = "cardinal" if xaxis_type == "index" else "ordinal"

        is_centred = self.centre is not None
        axes_titles = {
            "y": (
                "Explained variance ratio"
                if is_centred
                else "Proportion of total variation"
            ),
            "x": (
                f"Principal component {xaxis_type}"
                if is_centred
                else f"Component {xaxis_type}"
            ),
        }

        if n < s.axes_manager[-1].size:
            s = s.isig[:n]

        if fig is None:
            fig = plt.figure(**kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        if log:
            ax.set_yscale("log")

        if hline:
            ax.axhline(cutoff, linewidth=2, color="gray", linestyle="dashed", zorder=1)

        if vline:
            ax.axvline(
                index_number_significant_components,
                linewidth=2,
                color="gray",
                linestyle="dashed",
                zorder=1,
            )

        index_offset = 0
        if xaxis_type == "number":
            index_offset = 1

        if n_signal_pcs == n:
            ax.plot(
                range(index_offset, index_offset + n),
                s.isig[:n].data,
                **signal_fmt,
            )
        elif n_signal_pcs > 0:
            ax.plot(
                range(index_offset, index_offset + n_signal_pcs),
                s.isig[:n_signal_pcs].data,
                **signal_fmt,
            )
            ax.plot(
                range(index_offset + n_signal_pcs, index_offset + n),
                s.isig[n_signal_pcs:n].data,
                **noise_fmt,
            )
        else:
            ax.plot(
                range(index_offset, index_offset + n),
                s.isig[:n].data,
                **noise_fmt,
            )

        if xaxis_labeling == "cardinal":
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, p: hs_utils.ordinal(x))
            )

        ax.set_ylabel(axes_titles["y"])
        ax.set_xlabel(axes_titles["x"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.margins(0.05)
        ax.autoscale()
        ax.set_title(s.metadata.General.title, y=1.01)

        return ax

    def plot_explained_variance_ratio(self, *args, **kwargs):
        """Deprecated: use :meth:`plot_scree` instead."""
        warnings.warn(
            "plot_explained_variance_ratio() is deprecated, use plot_scree() instead.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        return self.plot_scree(*args, **kwargs)

    def plot_cumulative_scree(self, n=50):
        """Plot the cumulative scree plot up to *n* components.

        Parameters
        ----------
        n : int, default 50
            Number of components to show.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        target = self
        is_centred = target.centre is not None
        ylabel = (
            "Cumulative explained variance ratio"
            if is_centred
            else "Cumulative proportion of total variation"
        )
        xlabel = "Principal component" if is_centred else "Component"
        if n > target.explained_variance.shape[0]:
            n = target.explained_variance.shape[0]
        cumu = np.cumsum(target.explained_variance) / np.sum(target.explained_variance)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(n), cumu[:n])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def plot_cumulative_explained_variance_ratio(self, *args, **kwargs):
        """Deprecated: use :meth:`plot_cumulative_scree` instead."""
        warnings.warn(
            "plot_cumulative_explained_variance_ratio() is deprecated, "
            "use plot_cumulative_scree() instead.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )
        return self.plot_cumulative_scree(*args, **kwargs)

    def _get_number_significant_components(self):
        """Estimate the number of significant components via elbow detection."""
        if self.explained_variance_ratio is None:
            return None
        from hyperspy_ml.utils.preprocessing import estimate_elbow_position

        return int(estimate_elbow_position(self.explained_variance_ratio) + 1)

    # ------------------------------------------------------------------
    # Component manipulation
    # ------------------------------------------------------------------

    def normalize_decomposition_components(self, target="components", function=np.sum):
        """Normalize decomposition components.

        Each component (or score) is divided by ``function(target)``
        and the other matrix is multiplied by the same coefficient.

        Parameters
        ----------
        target : {"components", "scores", "factors", "loadings"}
            Which matrix to use for normalisation. ``"factors"`` and
            ``"loadings"`` are deprecated aliases.
        function : callable, default ``numpy.sum``
            Function applied along ``axis=0`` to produce the scale factor.
        """
        from hyperspy_ml.utils.preprocessing import _normalize_components

        if target == "factors":
            warnings.warn(
                '`target="factors"` is deprecated, use `target="components"` instead.',
                VisibleDeprecationWarning,
            )
            target_arr = self.components
            other = self.scores
        elif target == "loadings":
            warnings.warn(
                '`target="loadings"` is deprecated, use `target="scores"` instead.',
                VisibleDeprecationWarning,
            )
            target_arr = self.scores
            other = self.components
        elif target == "components":
            target_arr = self.components
            other = self.scores
        elif target == "scores":
            target_arr = self.scores
            other = self.components
        else:
            raise ValueError(
                'target must be "factors", "loadings", "components" or "scores"'
            )

        if target_arr is None:
            raise ValueError("This method requires components and scores to be set")

        _normalize_components(target=target_arr, other=other, function=function)
        self._notify_data_changed()

    def reverse_decomposition_component(self, component_number):
        """Reverse the sign of a decomposition component.

        Multiplies the specified column(s) of both ``components`` and
        ``scores`` by ``-1``.  Not supported for lazy (dask) results.

        Parameters
        ----------
        component_number : int or iterable of int
            Component index (or indices) to reverse.
        """
        if hasattr(self.components, "compute"):
            _logger.warning(
                f"Component(s) {component_number} not reversed, "
                "feature not implemented for lazy computations"
            )
            return

        for i in [component_number]:
            self.components[:, i] *= -1
            self.scores[:, i] *= -1
        self._notify_data_changed()


class _ResultEvents:
    """Minimal events namespace for result containers."""

    data_changed: bool = False

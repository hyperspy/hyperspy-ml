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

"""Signal-aware preprocessing helpers for MVA decomposition.

These helpers were originally defined inline in ``hyperspy_ml/_mva.py``.
They are extracted here to keep ``_mva.py`` focused on the MVA orchestration
class while still exposing the utility functions for external use.
"""

import importlib

import numpy as np
from hyperspy.misc import utils as hs_utils

# ---------------------------------------------------------------------------
# Algorithm lookup tables — used by the _get_sklearn_* helpers below.
# ---------------------------------------------------------------------------

decomposition_algorithms = {
    "sklearn_pca": "PCA",
    "NMF": "NMF",
    "sparse_pca": "SparsePCA",
    "mini_batch_sparse_pca": "MiniBatchSparsePCA",
    "sklearn_fastica": "FastICA",
}


cluster_algorithms = {
    None: "KMeans",
    "kmeans": "KMeans",
    "agglomerative": "AgglomerativeClustering",
    "minibatchkmeans": "MiniBatchKMeans",
    "spectralclustering": "SpectralClustering",
}


preprocessing_algorithms = {
    None: None,
    "norm": "Normalizer",
    "standard": "StandardScaler",
    "minmax": "MinMaxScaler",
}


# ---------------------------------------------------------------------------
# Scikit-learn algorithm lookup helpers
# ---------------------------------------------------------------------------


def _get_sklearn_algorithms(algorithm):
    """Get the sklearn algorithms available for decomposition."""
    module = importlib.import_module("sklearn.decomposition")
    return getattr(module, decomposition_algorithms[algorithm])


def _get_sklearn_clustering_algorithms(algorithm):
    """Get the sklearn algorithms available for clustering."""
    module = importlib.import_module("sklearn.cluster")
    return getattr(module, cluster_algorithms[algorithm])


def _get_sklearn_preprocessing_algorithms(algorithm):
    """Get the sklearn algorithms available for preprocessing."""
    module = importlib.import_module("sklearn.preprocessing")
    return getattr(module, preprocessing_algorithms[algorithm])


# ---------------------------------------------------------------------------
# Signal-aware derivative
# ---------------------------------------------------------------------------


def _get_derivative(signal, diff_axes, diff_order):
    """Calculate the derivative of a signal."""
    if signal.axes_manager.signal_dimension == 1:
        signal = signal.derivative(order=diff_order, axis=-1)
    else:
        # n-d signal case.
        # Compute the differences for each signal axis, unfold the
        # signal axes and stack the differences over the signal
        # axis.
        if diff_axes is None:
            diff_axes = signal.axes_manager.signal_axes
            iaxes = [axis.index_in_axes_manager for axis in diff_axes]
        else:
            iaxes = diff_axes
        diffs = [signal.derivative(order=diff_order, axis=i) for i in iaxes]
        for signal in diffs:
            signal.unfold()
        signal = hs_utils.stack(diffs, axis=-1)
        del diffs
    return signal


# ---------------------------------------------------------------------------
# NaN-expand rows (numpy + dask)
# ---------------------------------------------------------------------------


def _nan_expand_rows(arr, mask, total_rows):
    """Return *arr* expanded to *total_rows*, NaN at positions where *mask* is True.

    Works for both numpy and dask arrays.  ``mask`` is a flat boolean array of
    length ``total_rows``; rows where mask is True are NaN-filled and rows where
    mask is False are filled from ``arr`` in order.

    Parameters
    ----------
    arr : numpy.ndarray or dask.array.Array, shape (n_kept, n_components)
        The unmasked rows of the factor or loading matrix.
    mask : numpy.ndarray of bool, shape (total_rows,)
        True where the row was *excluded* from decomposition.
    total_rows : int
        Total number of rows in the expanded output (masked + unmasked).

    Returns
    -------
    numpy.ndarray or dask.array.Array, shape (total_rows, n_components)
    """
    try:
        import dask.array as _da
    except ImportError:
        _da = None

    unmasked_idx = np.where(~mask)[0]
    n_comp = arr.shape[1]

    if _da is not None and isinstance(arr, _da.Array):
        # Vectorized reindex via da.take() — avoids building one dask
        # task per row, which creates an enormous task graph and can
        # crash for large total_rows (e.g. millions of navigation pixels).
        row_idx = np.full(total_rows, -1, dtype=np.intp)
        row_idx[unmasked_idx] = np.arange(len(unmasked_idx), dtype=np.intp)
        # Pad arr with a NaN row so masked positions (index -1) map to NaN.
        nan_row_np = np.full((1, n_comp), np.nan, dtype=arr.dtype)
        arr_padded = _da.concatenate(
            [_da.from_array(nan_row_np, chunks=(1, n_comp)), arr], axis=0
        )
        # Shift indices: -1 → 0 (NaN row), 0 → 1, …
        take_idx = _da.from_array(row_idx + 1, chunks=(arr.chunks[0][0],))
        return _da.take(arr_padded, take_idx, axis=0)
    else:
        out = np.full((total_rows, n_comp), np.nan, dtype=arr.dtype)
        out[unmasked_idx, :] = arr
        return out


# ---------------------------------------------------------------------------
# Component normalisation
# ---------------------------------------------------------------------------


def _normalize_components(target, other, function=np.sum):
    """Normalize components according to a function."""
    coeff = function(target, axis=0)
    target /= coeff
    other *= coeff


# ---------------------------------------------------------------------------
# Mask normalisation (BaseSignal / dask / numpy → flat bool)
# ---------------------------------------------------------------------------


def _to_flat_bool(mask):
    """Return a flat boolean numpy array, ``True`` where *mask* is ``True``.

    Normalises mask inputs of various types (BaseSignal, dask array, numpy
    array, or ``None``) into a 1-D boolean numpy array suitable for
    boolean indexing and :func:`_nan_expand_rows`.

    Parameters
    ----------
    mask : BaseSignal, dask.array.Array, numpy.ndarray, or None
        The mask to normalise.  If ``None``, returns ``None``.

    Returns
    -------
    numpy.ndarray or None
        Flat boolean array where ``True`` = excluded position, or ``None``
        if *mask* was ``None``.
    """
    if mask is None:
        return None
    try:
        import dask.array as da
    except ImportError:
        da = None

    if da is not None and isinstance(mask, da.Array):
        mask = mask.compute()
    if hasattr(mask, "data"):
        mask = mask.data  # BaseSignal
    return np.asarray(mask, dtype=bool).ravel()


# ---------------------------------------------------------------------------
# Reprojection helpers
# ---------------------------------------------------------------------------


def _reproject_navigation_scores(D, components):
    """Compute full-navigation loadings via least-squares projection.

    Solves ``loadings = D @ components / ||components||^2``, which is the
    least-squares solution to ``components @ loadings ≈ D``.

    Parameters
    ----------
    D : ndarray or dask Array, shape (nav, sig)
        Data matrix.
    components : ndarray or dask Array, shape (sig, k)
        Factor matrix.

    Returns
    -------
    ndarray or dask Array, shape (nav, k)
        Loadings matrix.
    """
    s_sq = np.einsum("ij,ij->j", components, components)
    return (D @ components) / s_sq


def _reproject_signal_components(D, scores):
    """Compute full-signal factors via pseudo-inverse.

    Solves ``factors = (pinv(scores) @ D).T`` so the result has shape
    ``(sig, k)`` matching HyperSpy's factor convention (rows = signal
    channels, columns = components).

    Parameters
    ----------
    D : ndarray, shape (nav, sig)
        Data matrix.
    scores : ndarray, shape (nav, k)
        Loading matrix.

    Returns
    -------
    ndarray, shape (sig, k)
        Factor matrix.
    """
    return (np.linalg.pinv(scores) @ D).T


# ---------------------------------------------------------------------------
# Keenan-Kotula Poisson-noise scaling (numpy + dask)
# ---------------------------------------------------------------------------


def _keenan_kotula_scale(data, navigation_mask, signal_mask, ndim, sdim):
    """Apply Keenan-Kotula Poisson-noise scaling.

    Implements the variance-stabilising transform described in [Keenan2004]_:

        D_scaled[i,j] = D[i,j] / (sqrt(aG[i]) * sqrt(bH[j]))

    where ``aG[i]`` is the total counts for navigation position *i* (summed
    over unmasked signal channels) and ``bH[j]`` is the total counts for
    signal channel *j* (summed over unmasked navigation positions).

    Works for both numpy and dask arrays.  Masked positions contribute
    zero to the sums and their data values pass through unscaled.

    Parameters
    ----------
    data : ndarray or dask Array, shape (nav..., sig...)
        Data in NumPy axis order (navigation axes first, signal axes last).
        For 2-D data (*ndim* = 1, *sdim* = 1) the shape is ``(nav, sig)``.
    navigation_mask : ndarray or None, shape (nav...,)
        Boolean mask where ``True`` = exclude this navigation position.
        ``None`` means no navigation masking.
    signal_mask : ndarray or None, shape (sig...,)
        Boolean mask where ``True`` = exclude this signal channel.
        ``None`` means no signal masking.
    ndim : int
        Number of navigation dimensions (length of nav axes prefix).
    sdim : int
        Number of signal dimensions (length of sig axes suffix).

    Returns
    -------
    scaled_data : ndarray or dask Array
        Data after variance-stabilising scaling, same shape as *data*.
    sqrt_aG : ndarray or dask Array, shape (nav...,)
        Square root of total counts per navigation position.
    sqrt_bH : ndarray or dask Array, shape (sig...,)
        Square root of total counts per signal channel.

    Raises
    ------
    ValueError
        If negative values are found in the unmasked region or if all
        data are masked.

    References
    ----------
    .. [Keenan2004] M. Keenan and P. Kotula, "Accounting for Poisson
       noise in the multivariate analysis of ToF-SIMS spectrum images",
       Surf. Interface Anal 36(3) (2004): 203-212.
    """
    import numpy as _np

    # Pick the array backend based on the data type.  We must use the
    # native backend from the start because numpy ufuncs dispatch to
    # dask via __array_ufunc__, so _np.logical_not(dask_array) returns
    # a dask array.
    _is_dask = hasattr(data, "chunks")
    if _is_dask:
        import dask.array as _da

        _lib = _da
        _nav_chunks = data.chunks[:ndim]
        _sig_chunks = data.chunks[ndim:]
    else:
        _lib = _np

    # "Keep" masks (True = use this position).
    if navigation_mask is None:
        nm = (
            _lib.ones(data.shape[:ndim], dtype=bool, chunks=_nav_chunks)
            if _is_dask
            else _lib.ones(data.shape[:ndim], dtype=bool)
        )
    else:
        nm = _lib.logical_not(navigation_mask)
    if signal_mask is None:
        sm = (
            _lib.ones(data.shape[ndim:], dtype=bool, chunks=_sig_chunks)
            if _is_dask
            else _lib.ones(data.shape[ndim:], dtype=bool)
        )
    else:
        sm = _lib.logical_not(signal_mask)

    # Broadcast to full data shape: nm → (nav..., 1...), sm → (1..., sig...)
    nm_bc = nm[(...,) + (None,) * sdim]
    sm_bc = sm[(None,) * ndim + (...,)]
    combined = nm_bc & sm_bc

    # Zero out masked entries so they do not contribute to aG / bH.
    masked_data = _lib.where(combined, data, 0.0)

    # Guard against negative values in the unmasked region.
    min_val = masked_data.min()
    if hasattr(min_val, "compute"):
        min_val = min_val.compute()
    if min_val < 0.0:
        raise ValueError(
            "Negative values found in data!\n"
            "Are you sure that the data follow a Poisson distribution?"
        )

    nav_axes = tuple(range(ndim, ndim + sdim))
    sig_axes = tuple(range(ndim))

    aG = masked_data.sum(axis=nav_axes)
    bH = masked_data.sum(axis=sig_axes)

    # Materialise sums for dask; check the zero-sum guard on the computed
    # NumPy values before re-wrapping, because float(dask_array) is not
    # guaranteed to work across dask versions.
    if _is_dask:
        aG, bH = (aG.compute(), bH.compute())

    # aG, bH are now guaranteed numpy arrays — float() below is safe.
    # The zero-sum guard must run on NumPy values, so re-wrapping aG/bH
    # as dask arrays is deliberately delayed until after this check.
    if float(aG.sum()) == 0.0:
        raise ValueError("All the data are masked, change the mask.")

    # Re-wrap as dask so downstream sqrt / broadcast stays lazy.
    if _is_dask:
        aG = _da.from_array(aG)
        bH = _da.from_array(bH)

    # Avoid division-by-zero for masked positions (they contribute 0 to
    # the sum so sqrt(0) would be zero — replace with 1 instead).
    aG = _lib.where(aG == 0, 1, aG)
    bH = _lib.where(bH == 0, 1, bH)

    sqrt_aG = _lib.sqrt(aG)
    sqrt_bH = _lib.sqrt(bH)

    coeff = sqrt_aG[(...,) + (None,) * sdim] * sqrt_bH[(None,) * ndim + (...,)]
    scaled_data = _lib.where(combined, data / coeff, data)

    return scaled_data, sqrt_aG, sqrt_bH


# ---------------------------------------------------------------------------
# Elbow-position estimation (extracted from MVA.estimate_elbow_position)
# ---------------------------------------------------------------------------


def estimate_elbow_position(explained_variance_ratio, log=True, max_points=20):
    """Estimate the elbow position of a scree plot curve.

    Used to estimate the number of significant components in
    a PCA variance ratio plot or other "elbow" type curves.

    Find a line between first and last point on the scree plot.
    With a classic elbow scree plot, this line more or less
    defines a triangle. The elbow should be the point which
    is the furthest distance from this line. For more details,
    see [1]_.

    Parameters
    ----------
    explained_variance_ratio : numpy array
        Explained variance ratio values that form the scree plot.
    max_points : int
        Maximum number of points to consider in the calculation.
    log : bool, default True
        If True, compute distances in log space.

    Returns
    -------
    int
        The index of the elbow position in the input array. Due to
        zero-based indexing, the number of significant components
        is ``elbow_position + 1``.

    References
    ----------
    .. [1] V. Satopää, J. Albrecht, D. Irwin, and B. Raghavan.
       "Finding a "Kneedle" in a Haystack: Detecting Knee Points in
       System Behavior,. 31st International Conference on Distributed
       Computing Systems Workshops, pp. 166-171, June 2011.
    """
    max_points = min(max_points, len(explained_variance_ratio) - 1)
    # Clipping the curve_values from below with a v.small
    # number avoids warnings below when taking np.log(0)
    curve_values_adj = np.clip(explained_variance_ratio, 1e-30, None)

    x1 = 0
    x2 = max_points

    if log:
        y1 = np.log(curve_values_adj[0])
        y2 = np.log(curve_values_adj[max_points])
    else:
        y1 = curve_values_adj[0]
        y2 = curve_values_adj[max_points]

    xs = np.arange(max_points, like=explained_variance_ratio)
    if log:
        ys = np.log(curve_values_adj[:max_points])
    else:
        ys = curve_values_adj[:max_points]

    numer = abs((x2 - x1) * (y1 - ys) - (x1 - xs) * (y2 - y1))
    denom = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = np.nan_to_num(numer / denom)
    elbow_position = np.argmax(distance)

    return elbow_position


# ---------------------------------------------------------------------------
# Convenience wrappers for the full preprocessing pipeline
# ---------------------------------------------------------------------------


def center_data(data, centre, ndim=1):
    """Center *data* along navigation or signal axes and return the mean.

    Parameters
    ----------
    data : ndarray, shape (nav..., sig...)
        Data to centre.  Navigation axes come first, signal axes last.
    centre : {None, "navigation", "signal"}
        If ``None``, return the data unchanged with ``mean = None``.
        If ``"navigation"``, subtract the mean over navigation positions.
        If ``"signal"``, subtract the mean over signal channels.
    ndim : int, default 1
        Number of navigation dimensions.  Signal dimensions are the
        remaining ``data.ndim - ndim`` axes.

    Returns
    -------
    centred : ndarray
        Centred data (same shape as input, or view if no centering).
    mean : ndarray or None
        The subtracted mean, or ``None`` when ``centre`` is ``None``.
    """
    if centre is None:
        return data, None

    nav_axes = tuple(range(ndim))
    sig_axes = tuple(range(ndim, data.ndim))

    if centre == "navigation":
        mean = data.mean(axis=nav_axes, keepdims=True)
    elif centre == "signal":
        mean = data.mean(axis=sig_axes, keepdims=True)
    else:
        raise ValueError(
            f"`centre` must be None, 'navigation' or 'signal', not {centre!r}"
        )

    centred = data - mean
    return centred, mean


def apply_preprocessing(
    data,
    centre=None,
    navigation_mask=None,
    signal_mask=None,
    normalize_poissonian_noise=False,
    ndim=1,
    sdim=1,
):
    """Run the full MVA preprocessing pipeline on a data matrix.

    Applies Keenan-Kotula Poisson-noise scaling (optional) followed by
    optional centering.

    Parameters
    ----------
    data : ndarray or dask Array, shape (nav..., sig...)
        Data in NumPy axis order (navigation axes first, signal axes last).
    centre : {None, "navigation", "signal"}, default None
        Centering mode.  ``None`` skips centering.
    navigation_mask : ndarray or None, shape (nav...,)
        Boolean mask where ``True`` = exclude position.  Only used when
        ``normalize_poissonian_noise`` is ``True``.
    signal_mask : ndarray or None, shape (sig...,)
        Boolean mask where ``True`` = exclude position.  Only used when
        ``normalize_poissonian_noise`` is ``True``.
    normalize_poissonian_noise : bool, default False
        If ``True``, apply Keenan-Kotula scaling before centering.
    ndim : int, default 1
        Number of navigation dimensions.
    sdim : int, default 1
        Number of signal dimensions.

    Returns
    -------
    processed : ndarray or dask Array
        Preprocessed data (same shape as input).
    mean : ndarray or None
        The subtracted mean (numpy array), or ``None`` if not centred.
    sqrt_aG : ndarray or None
        Navigation weighting factors from K-K scaling, or ``None``.
    sqrt_bH : ndarray or None
        Signal weighting factors from K-K scaling, or ``None``.
    """
    sqrt_aG = None
    sqrt_bH = None

    if normalize_poissonian_noise:
        if centre is not None:
            raise ValueError(
                "normalize_poissonian_noise=True is only compatible "
                f"with `centre=None`, not `centre={centre}`."
            )
        data, sqrt_aG, sqrt_bH = _keenan_kotula_scale(
            data, navigation_mask, signal_mask, ndim, sdim
        )

    processed, mean_val = center_data(data, centre, ndim=ndim)

    return processed, mean_val, sqrt_aG, sqrt_bH

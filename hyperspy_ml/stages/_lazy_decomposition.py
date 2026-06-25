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

"""Lazy (dask-backed) decomposition helpers for the Decomposition stage.

These functions are adapted from :class:`hyperspy._signals.lazy.LazySignal`
(decomposition, ``_decomposition_svd_matrix``, ``_decomposition_reproject_signal``,
``_decomposition_reproject_navigation``, and ``_project_scores``).

The original per-line git history is lost because these methods were embedded in
the ``LazySignal`` class (~751 lines) and cannot be extracted via ``git filter-repo``.
"""

from __future__ import annotations

import warnings
from itertools import product

import numpy as np
from hyperspy.exceptions import VisibleDeprecationWarning


def _iterate_dask_chunks(data, navigation_indices, flat_signal=True):
    """Yield navigation chunks from a dask array as numpy arrays.

    Iterates over the navigation dimensions of *data*, computing one
    chunk at a time.  This is the stage-equivalent of
    :meth:`LazySignal._block_iterator`, adapted to operate on raw dask
    arrays rather than HyperSpy signal instances.

    Parameters
    ----------
    data : dask.array.Array, shape (nav..., sig...)
        The dask array to iterate over.
    navigation_indices : list of range
        Ranges over the navigation chunk indices, typically obtained via
        ``[range(len(c)) for c in data.chunks[:nav_dim]]``.
    flat_signal : bool, default True
        If True, flatten the signal dimensions so each chunk is 2-D
        ``(chunk_nav_size, signal_size)``.
        If False, return the chunk in the original N-D shape.

    Yields
    ------
    numpy.ndarray
        One computed navigation chunk.
    """

    nav_dim = len(navigation_indices)
    sig_dim = data.ndim - nav_dim
    signalsize = (
        int(np.prod(data.shape[nav_dim:])) if flat_signal and sig_dim > 0 else 0
    )

    for ind in product(*navigation_indices):
        chunk = data.blocks[ind + (0,) * sig_dim]
        computed = chunk.compute()
        if flat_signal and signalsize > 0:
            computed = computed.reshape(-1, signalsize)
        yield computed


def _resolve_mask_to_flat_bool(mask):
    """Resolve a mask to a flat boolean numpy array.

    Handles BaseSignal, dask, and numpy masks.  Returns ``None`` when
    *mask* is ``None``.

    Parameters
    ----------
    mask : BaseSignal, dask.array.Array, numpy.ndarray, or None

    Returns
    -------
    numpy.ndarray or None
        Flat boolean array where True = excluded position.
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
        mask = mask.data
    return np.asarray(mask, dtype=bool).ravel()


def _lazy_svd_matrix(
    data,
    svd_solver,
    centre,
    output_dimension,
    navigation_mask,
    signal_mask,
):
    """Run SVD on a 2-D dask data matrix.

    Unfolds the signal, resolves navigation and signal masks to 1-D
    boolean arrays, applies centring if requested, computes the SVD
    via either ``da.linalg.svd`` (full) or ``da.linalg.svd_compressed``
    (randomized).

    This is adapted from :meth:`LazySignal._decomposition_svd_matrix`.

    Parameters
    ----------
    data : dask.array.Array, shape (nav, sig)
        The 2-D (unfolded) data matrix.
    svd_solver : str
        ``"full"`` or ``"randomized"``.
    centre : str or None
        Centring strategy (``"navigation"``, ``"signal"``, or ``None``).
    output_dimension : int or None
        Number of components; required for ``"randomized"``.
    navigation_mask : numpy.ndarray, dask.array.Array, or None
        Navigation mask (True = excluded).
    signal_mask : numpy.ndarray, dask.array.Array, or None
        Signal mask (True = excluded).

    Returns
    -------
    dict
        Keys: ``scores``, ``components``, ``explained_variance``, ``mean``,
        ``nav_mask_1d``, ``sig_mask_1d``, ``_nav_mask_for_reproject``,
        ``_sig_mask_for_reproject``.
    """
    import dask
    import dask.array as da

    # Resolve navigation mask to a 1-D boolean numpy array.
    nav_mask_1d = None
    _nav_mask_for_reproject = navigation_mask
    if navigation_mask is not None:
        if hasattr(navigation_mask, "data"):
            _nm = navigation_mask.data
        else:
            _nm = navigation_mask
        _nav_mask_for_reproject = _nm
        if isinstance(_nm, da.Array):
            nav_mask_1d = _nm.ravel().compute().astype(bool)
        else:
            nav_mask_1d = np.asarray(_nm).ravel().astype(bool)

    # Resolve signal mask to a 1-D boolean numpy array.
    sig_mask_1d = None
    _sig_mask_for_reproject = signal_mask
    if signal_mask is not None:
        if hasattr(signal_mask, "data"):
            _sm = signal_mask.data
        else:
            _sm = signal_mask
        _sig_mask_for_reproject = _sm
        if isinstance(_sm, da.Array):
            sig_mask_1d = _sm.ravel().compute().astype(bool)
        else:
            sig_mask_1d = np.asarray(_sm).ravel().astype(bool)

    # Build the data matrix, applying masks if present.
    D = data
    if nav_mask_1d is not None:
        D = D[~nav_mask_1d, :]
    if sig_mask_1d is not None:
        D = D[:, ~sig_mask_1d]

    if svd_solver == "full":
        warnings.warn(
            "svd_solver='full' is deprecated and will be removed in "
            "HyperSpy 3.0.  Use svd_solver='randomized' instead, "
            "which gives identical results for truncated SVD with "
            "substantially lower memory usage.",
            VisibleDeprecationWarning,
            stacklevel=2,
        )

        if centre == "navigation":
            mean = D.mean(axis=0, keepdims=True).compute()
            D = D - mean
        elif centre == "signal":
            mean = D.mean(axis=1, keepdims=True).compute()
            D = D - mean
        else:
            mean = None

        if D.numblocks[1] > 1:
            D = D.rechunk({1: -1})
        U, S, V = da.linalg.svd(D)
        if output_dimension is not None:
            U = U[:, :output_dimension]
            S = S[:output_dimension]
            V = V[:output_dimension]
        components = V.T  # dask array — lazy
        explained_variance = S**2 / D.shape[0]
        scores = U * S  # dask array — lazy
    else:  # randomized
        if centre == "navigation":
            mean = D.mean(axis=0, keepdims=True).compute()
            D = D - mean
        elif centre == "signal":
            mean = D.mean(axis=1, keepdims=True).compute()
            D = D - mean
        else:
            mean = None

        # Use the synchronous scheduler for svd_compressed to
        # avoid materialising the full dataset in memory.
        with dask.config.set(scheduler="synchronous"):
            U, S, V = da.linalg.svd_compressed(D, k=output_dimension)
            U, S, V = dask.compute(U, S, V)

        components = V.T  # (n_unmasked_sig, output_dimension) — numpy
        explained_variance = S**2 / D.shape[0]
        scores = U * S

    # Build masked nav mask for reprojection use — block_iterator expects
    # a shaped mask.  For the lazy stage we use the flat masks since we
    # don't have block_iterator.
    return {
        "scores": scores,
        "components": components,
        "explained_variance": explained_variance,
        "mean": mean,
        "nav_mask_1d": nav_mask_1d,
        "sig_mask_1d": sig_mask_1d,
        "_nav_mask_for_reproject": _nav_mask_for_reproject,
        "_sig_mask_for_reproject": _sig_mask_for_reproject,
    }


def _lazy_reproject_navigation(
    reproject,
    algorithm,
    svd_solver,
    scores,
    components,
    mean,
    centre,
    data_unfolded,
    sig_mask_1d,
):
    """Reproject scores over the full (unmasked) navigation space.

    Uses dask matmuls for ``svd_solver='full'`` and ``'randomized'``,
    streaming over nav chunks without materialising the full matrix.

    Adapted from :meth:`LazySignal._decomposition_reproject_navigation`.

    Parameters
    ----------
    reproject : str or None
    algorithm : str
    svd_solver : str
    scores : ndarray or dask Array
    components : ndarray or dask Array
    mean : ndarray or None
    centre : str or None
    data_unfolded : dask Array, shape (nav, sig)
    sig_mask_1d : ndarray or None

    Returns
    -------
    scores : ndarray or dask Array
    _nav_reprojected : bool
    """
    import dask.array as da

    from hyperspy_ml.utils.preprocessing import _reproject_navigation_scores

    _nav_reprojected = False
    if reproject in ("navigation", "both"):
        if algorithm == "SVD" and svd_solver in ("full", "randomized"):
            D_nav = data_unfolded
            if sig_mask_1d is not None:
                D_nav = D_nav[:, ~sig_mask_1d]
            if mean is not None and centre == "navigation":
                D_nav = D_nav - mean
            _components_da = (
                components
                if isinstance(components, da.Array)
                else da.from_array(components)
            )
            # Dask matmul — streams over nav chunks.
            scores = _reproject_navigation_scores(D_nav, _components_da).compute()
        elif reproject == "navigation":
            # For non-SVD algorithms we compute the least-squares scores
            # via pinv-L @ D.  The data is dask; compute incrementally.
            from hyperspy_ml.utils.preprocessing import _reproject_navigation_scores

            D_nav = data_unfolded
            if sig_mask_1d is not None:
                D_nav = D_nav[:, ~sig_mask_1d]
            if mean is not None and centre == "navigation":
                D_nav = D_nav - mean
            _components_da = (
                components
                if isinstance(components, da.Array)
                else da.from_array(components)
            )
            scores = _reproject_navigation_scores(D_nav, _components_da).compute()
        _nav_reprojected = True
    return scores, _nav_reprojected


def _lazy_reproject_signal(
    algorithm,
    svd_solver,
    reproject,
    scores,
    components,
    mean,
    centre,
    data_unfolded,
    nav_mask_1d,
    flat_sig_mask,
    flat_nav_mask,
):
    """Reproject components over the full (unmasked) signal space.

    Uses dask matmuls to stream over nav chunks for
    ``svd_solver='full'``.  For other solvers, collects each nav chunk
    and solves ``pinv(L) @ D`` eagerly.

    Adapted from :meth:`LazySignal._decomposition_reproject_signal`.

    Parameters
    ----------
    algorithm : str
    svd_solver : str
    reproject : str
    scores : ndarray or dask Array
    components : ndarray or dask Array
    mean : ndarray or None
    centre : str or None
    data_unfolded : dask Array, shape (nav, sig)
    nav_mask_1d : ndarray or None
    flat_sig_mask : ndarray or None
    flat_nav_mask : ndarray or None

    Returns
    -------
    components : ndarray
    """
    import dask.array as da

    from hyperspy_ml.utils.preprocessing import _reproject_signal_components

    if algorithm == "SVD" and svd_solver == "full":
        D_sig = data_unfolded
        if nav_mask_1d is not None:
            D_sig = D_sig[~nav_mask_1d, :]
        if mean is not None and centre == "navigation":
            D_sig = D_sig - mean
        if reproject == "both":
            if flat_nav_mask is not None:
                L = scores[~flat_nav_mask, :]
            else:
                L = scores
        else:
            L = scores  # already unmasked-nav only
        pinv_L = np.linalg.pinv(L.compute() if isinstance(L, da.Array) else L)
        # (k, n_unmasked_nav) @ (n_unmasked_nav, sig) = (k, sig)
        # Dask streams over nav chunks; result is small.
        components = (da.from_array(pinv_L) @ D_sig).T.compute()
    else:
        # For non-full SVD or non-SVD algorithms, collect chunks
        # and solve eagerly.
        D_sig = data_unfolded
        if nav_mask_1d is not None:
            D_sig = D_sig[~nav_mask_1d, :]
        # Compute the full (nav, sig) data
        D = D_sig.compute()
        if mean is not None:
            mean_1d = np.asarray(mean).ravel()
            if flat_sig_mask is not None and len(mean_1d) < D.shape[1]:
                mean_full = np.zeros(D.shape[1], dtype=mean_1d.dtype)
                mean_full[~flat_sig_mask] = mean_1d
                D = D - mean_full
            else:
                D = D - mean_1d
        if reproject == "both":
            if flat_nav_mask is not None:
                L = scores[~flat_nav_mask, :]
            else:
                L = scores
        else:
            L = scores
        components = _reproject_signal_components(D, L)
    return components


def _nan_expand_rows_lazy(arr, mask, total_rows):
    """Return *arr* expanded to *total_rows*, NaN at masked positions.

    Dask-aware variant of :func:`_nan_expand_rows`.
    Works for both numpy and dask arrays.

    Parameters
    ----------
    arr : ndarray or dask Array, shape (n_kept, n_components)
    mask : ndarray of bool, shape (total_rows,)
    total_rows : int

    Returns
    -------
    ndarray or dask Array, shape (total_rows, n_components)
    """
    try:
        import dask.array as da
    except ImportError:
        da = None

    unmasked_idx = np.where(~mask)[0]
    n_comp = arr.shape[1]

    if da is not None and isinstance(arr, da.Array):
        row_idx = np.full(total_rows, -1, dtype=np.intp)
        row_idx[unmasked_idx] = np.arange(len(unmasked_idx), dtype=np.intp)
        nan_row_np = np.full((1, n_comp), np.nan, dtype=arr.dtype)
        arr_padded = da.concatenate(
            [da.from_array(nan_row_np, chunks=(1, n_comp)), arr], axis=0
        )
        take_idx = da.from_array(row_idx + 1, chunks=(arr.chunks[0][0],))
        return da.take(arr_padded, take_idx, axis=0)
    else:
        out = np.full((total_rows, n_comp), np.nan, dtype=arr.dtype)
        out[unmasked_idx, :] = arr
        return out

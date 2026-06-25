# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Convenience functions for the hyperspy-ml public API.

Thin wrappers that construct stages, run fit_transform, and return results.
Matches the Decomposition/BSS/Clustering stage paths from Tasks 4-5.
"""

from __future__ import annotations

from typing import Any

from hyperspy_ml.results.base import BSSResult, ClusterResult, DecompositionResult
from hyperspy_ml.stages.bss import BSS
from hyperspy_ml.stages.clustering import Clustering
from hyperspy_ml.stages.decomposition import Decomposition


def decompose(
    signal,
    normalize_poissonian_noise: bool = False,
    algorithm: str | object = "SVD",
    output_dimension: int | None = None,
    centre: str | None = None,
    navigation_mask: Any = None,
    signal_mask: Any = None,
    reproject: str | None = None,
    print_info: bool = False,
    svd_solver: str = "auto",
    num_chunks: int | None = None,
    **kwargs,
) -> DecompositionResult:
    """Run decomposition on *signal* and return a DecompositionResult.

    Thin wrapper around ``Decomposition(...).fit_transform(signal)``.
    Parameters match the legacy ``signal.decomposition()`` signature.
    """
    stage = Decomposition(
        normalize_poissonian_noise=normalize_poissonian_noise,
        algorithm=algorithm,
        output_dimension=output_dimension,
        centre=centre,
        navigation_mask=navigation_mask,
        signal_mask=signal_mask,
        reproject=reproject,
        print_info=print_info,
        svd_solver=svd_solver,
        num_chunks=num_chunks,
        **kwargs,
    )
    result = stage.fit_transform(signal)
    result._nav_shape = signal.axes_manager.navigation_shape[::-1]
    result._source_signal = signal
    return result


def bss(
    decomposition_result: DecompositionResult,
    number_of_components: int | None = None,
    algorithm: str | object = "orthomax",
    on_scores: bool = False,
    print_info: bool = False,
    reverse_component_criterion: str = "components",
    whiten_method: str | None = "PCA",
    **kwargs,
) -> tuple[BSSResult, Any]:
    """Run BSS on a decomposition result.

    Thin wrapper around ``BSS(...).fit_transform(result)``.
    """
    stage = BSS(
        number_of_components=number_of_components,
        algorithm=algorithm,
        on_scores=on_scores,
        print_info=print_info,
        reverse_component_criterion=reverse_component_criterion,
        whiten_method=whiten_method,
        **kwargs,
    )
    return stage.fit_transform(decomposition_result)


def cluster(
    decomposition_result: DecompositionResult,
    signal=None,
    n_clusters: int | None = None,
    cluster_source: str | Any = "decomposition",
    algorithm: str | object | None = None,
    preprocessing: str | object | None = None,
    print_info: bool = False,
    **kwargs,
) -> ClusterResult:
    """Run clustering on a decomposition result.

    Thin wrapper around ``Clustering(...).fit_transform(result, signal)``.
    """
    stage = Clustering(
        n_clusters=n_clusters,
        cluster_source=cluster_source,
        algorithm=algorithm,
        preprocessing=preprocessing,
        print_info=print_info,
        **kwargs,
    )
    return stage.fit_transform(decomposition_result, signal=signal)


def load_result(path):
    """Load a .hsml or legacy .npz result file.

    Dispatches to :func:`hyperspy_ml.results.io.load_result` or
    :func:`hyperspy_ml.results.io.load_npz` based on file extension.
    """
    from pathlib import Path

    from hyperspy_ml.results.io import extract_results

    return extract_results(Path(path))


def extract_results(source):
    """Extract results from a signal or file.

    Parameters
    ----------
    source : BaseSignal, str, or Path

    Returns
    -------
    DecompositionResult
    """
    from hyperspy_ml.results.io import extract_results as _extract

    return _extract(source)

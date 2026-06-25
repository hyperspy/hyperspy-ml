# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Convenience functions bridging old signal-based API to new stage-based API.

These wrappers enable gradual porting of old-style tests that call
``signal.decomposition(...)``, ``signal.blind_source_separation(...)``, etc.
"""

from __future__ import annotations

from hyperspy_ml.results.base import BSSResult, ClusterResult, DecompositionResult
from hyperspy_ml.stages.bss import BSS
from hyperspy_ml.stages.clustering import Clustering
from hyperspy_ml.stages.decomposition import Decomposition


def decompose(
    signal,
    normalize_poissonian_noise=False,
    algorithm="SVD",
    output_dimension=None,
    centre=None,
    navigation_mask=None,
    signal_mask=None,
    reproject=None,
    print_info=False,
    svd_solver="auto",
    num_chunks=None,
    **kwargs,
) -> DecompositionResult:
    """Run decomposition on *signal* and return a DecompositionResult.

    Convenience wrapper for::

        Decomposition(...).fit_transform(signal)

    Parameters match the old signal.decomposition() signature.
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
    # Store the source signal on the result for model reconstruction
    result._nav_shape = signal.axes_manager.navigation_shape[::-1]
    result._source_signal = signal
    return result


def bss_analysis(
    decomposition_result,
    number_of_components=None,
    algorithm="orthomax",
    on_scores=False,
    print_info=False,
    reverse_component_criterion="components",
    whiten_method="PCA",
    **kwargs,
) -> BSSResult:
    """Run BSS on *decomposition_result*.

    Returns BSSResult (ignore return_info second element).
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
    bss_result, _ = stage.fit_transform(decomposition_result)
    return bss_result


def cluster_analysis(
    decomposition_result,
    signal=None,
    n_clusters=None,
    cluster_source="decomposition",
    algorithm=None,
    preprocessing=None,
    print_info=False,
    **kwargs,
) -> ClusterResult:
    """Run clustering on *decomposition_result*.

    Returns ClusterResult.
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

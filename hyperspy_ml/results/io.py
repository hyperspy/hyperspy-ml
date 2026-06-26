# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Zarr-based .hsml I/O with type-tag dispatch and legacy .npz reader.

Provides the main save/load entry points for the hyperspy-ml ecosystem,
dispatched by result type.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

HSML_VERSION = "1.0"

_RESULT_TYPE_MAP: dict[str, type] = {}


def _register_type(name, cls):
    _RESULT_TYPE_MAP[name] = cls
    _RESULT_TYPE_MAP[cls.__name__] = cls


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_result(result, path, source_signal=None):
    """Save a result object as a .hsml Zarr store.

    Parameters
    ----------
    result : DecompositionResult, BSSResult, or ClusterResult
    path : str or Path
    source_signal : BaseSignal or None
    """
    import zarr

    store = zarr.open(str(path), mode="w")
    store.attrs["hsml_version"] = HSML_VERSION
    store.attrs["type"] = type(result).__name__

    # Array fields as Zarr arrays
    for name in _array_fields(type(result)):
        value = getattr(result, name, None)
        if value is not None:
            store[name] = np.asarray(value)

    # Scalar fields as JSON
    scalars = {}
    for name in _scalar_fields(type(result)):
        value = getattr(result, name, None)
        if value is not None:
            scalars[name] = value
    store.attrs["scalars"] = json.dumps(scalars)

    if result.params:
        store.attrs["params"] = json.dumps(result.params)

    sig = source_signal or getattr(result, "_source_signal", None)
    if sig is not None:
        store.attrs["axes"] = json.dumps(
            {
                "navigation_shape": sig.axes_manager.navigation_shape,
                "signal_shape": sig.axes_manager.signal_shape,
            }
        )

    if sig is not None:
        src_hash = _source_hash(sig)
        result._source_hash = src_hash
        store.attrs["source_hash"] = src_hash
        src_grp = store.require_group("sources")
        if src_hash not in src_grp:
            src_val = sig.data
            if hasattr(src_val, "compute"):
                src_val = src_val.compute()
            src_grp[src_hash] = np.asarray(src_val)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_result(path):
    """Load a result object from a .hsml Zarr store.

    Raises OSError on major version mismatch.
    """
    import zarr

    store = zarr.open(str(path), mode="r")
    file_ver = store.attrs.get("hsml_version", "0.0")
    if file_ver.split(".")[0] != HSML_VERSION.split(".")[0]:
        raise OSError(
            f"Cannot load .hsml file: version {file_ver} "
            f"incompatible with {HSML_VERSION}."
        )

    type_name = store.attrs.get("type", "DecompositionResult")
    cls = _RESULT_TYPE_MAP.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown result type '{type_name}'. Known: {list(_RESULT_TYPE_MAP.keys())}"
        )

    kwargs: dict[str, Any] = {}
    for name in _array_fields(cls):
        if name in store:
            kwargs[name] = store[name][:]

    scalars = json.loads(store.attrs.get("scalars", "{}"))
    kwargs.update(scalars)
    kwargs["params"] = json.loads(store.attrs.get("params", "{}"))
    kwargs["_source_hash"] = store.attrs.get("source_hash", None)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Legacy .npz reader
# ---------------------------------------------------------------------------


def load_npz(path):
    """Load a legacy .npz decomposition file into a DecompositionResult.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    DecompositionResult
    """
    from hyperspy_ml.results.base import DecompositionResult
    from hyperspy_ml.results.learning_results import LearningResults

    lr = LearningResults()
    lr.load(str(path))

    return DecompositionResult(
        components=lr.components,
        scores=lr.scores,
        explained_variance=lr.explained_variance,
        explained_variance_ratio=lr.explained_variance_ratio,
        mean=lr.mean,
        algorithm=lr.decomposition_algorithm,
        output_dimension=lr.output_dimension,
        centre=lr.centre,
        poissonian_noise_normalized=bool(lr.poissonian_noise_normalized),
        number_significant_components=lr.number_significant_components,
        navigation_mask=lr.navigation_mask,
        signal_mask=lr.signal_mask,
        bss_components=lr.bss_components,
        bss_scores=lr.bss_scores,
        unmixing_matrix=lr.unmixing_matrix,
        bss_algorithm=lr.bss_algorithm,
        params={"loaded_from": str(path), "format": "npz"},
    )


# ---------------------------------------------------------------------------
# extract_results — pre-3.0 file migration
# ---------------------------------------------------------------------------


def extract_results(source):
    """Extract results from a pre-3.0 signal or file.

    Parameters
    ----------
    source : BaseSignal, str, or Path
        If a signal, extract from ``signal.learning_results``.
        If a path, load as .hsml or legacy .npz.

    Returns
    -------
    DecompositionResult
    """
    from pathlib import Path

    if isinstance(source, (str, Path)):
        path = str(source)
        if path.endswith(".npz"):
            return load_npz(path)
        return load_result(path)

    # Signal → extract from learning_results
    lr = source.learning_results
    from hyperspy_ml.results.base import DecompositionResult

    def _get_or_none(obj, *attrs):
        for a in attrs:
            val = getattr(obj, a, None)
            if val is not None:
                return val
        return None

    return DecompositionResult(
        components=_get_or_none(lr, "components", "factors"),
        scores=_get_or_none(lr, "scores", "loadings"),
        explained_variance=getattr(lr, "explained_variance", None),
        explained_variance_ratio=getattr(lr, "explained_variance_ratio", None),
        mean=getattr(lr, "mean", None),
        algorithm=getattr(lr, "decomposition_algorithm", None),
        output_dimension=getattr(lr, "output_dimension", None),
        centre=getattr(lr, "centre", None),
        poissonian_noise_normalized=bool(
            getattr(lr, "poissonian_noise_normalized", False)
        ),
        number_significant_components=getattr(
            lr, "number_significant_components", None
        ),
        navigation_mask=getattr(lr, "navigation_mask", None),
        signal_mask=getattr(lr, "signal_mask", None),
        bss_components=_get_or_none(lr, "bss_components", "bss_factors"),
        bss_scores=_get_or_none(lr, "bss_scores", "bss_loadings"),
        unmixing_matrix=getattr(lr, "unmixing_matrix", None),
        bss_algorithm=getattr(lr, "bss_algorithm", None),
        params={"extracted_from_signal": True},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_hash(signal) -> str:
    import hashlib

    data = signal.data
    if hasattr(data, "compute"):
        data = data.compute()
    return hashlib.sha256(np.asarray(data, dtype=np.float64).tobytes()).hexdigest()[:16]


def _array_fields(cls):
    return [
        f.name for f in cls.__dataclass_fields__.values() if "ndarray" in str(f.type)
    ]


def _scalar_fields(cls):
    return [
        f.name
        for f in cls.__dataclass_fields__.values()
        if f.name not in _array_fields(cls)
        and not f.name.startswith("_")
        and f.name != "params"
        and "ndarray" not in str(f.type)
        and "dict" not in str(f.type).lower()
    ]


# Register built-in types
def _init_registry():
    from hyperspy_ml.results.base import BSSResult, ClusterResult, DecompositionResult

    for cls in (DecompositionResult, BSSResult, ClusterResult):
        _register_type(cls.__name__, cls)


_init_registry()

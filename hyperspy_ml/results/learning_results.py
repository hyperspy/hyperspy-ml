# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
"""Standalone LearningResults with .npz backward-compatible save/load.

This is a copy of the legacy ``hyperspy.learn._mva.LearningResults`` class
updated to use the canonical ``components``/``scores`` terminology.
Includes deprecated property aliases (factors/loadings) for backward compat.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from hyperspy.exceptions import VisibleDeprecationWarning
from rsciio.utils import path as rs_path

_logger = logging.getLogger(__name__)

_SAVEABLE_ATTRS = (
    "components",
    "scores",
    "explained_variance",
    "explained_variance_ratio",
    "number_significant_components",
    "decomposition_algorithm",
    "poissonian_noise_normalized",
    "output_dimension",
    "mean",
    "centre",
    "cluster_membership",
    "cluster_labels",
    "cluster_centers",
    "cluster_centers_estimated",
    "cluster_algorithm",
    "number_of_clusters",
    "estimated_number_of_clusters",
    "cluster_metric_data",
    "cluster_metric_index",
    "cluster_metric",
    "bss_algorithm",
    "unmixing_matrix",
    "bss_components",
    "bss_scores",
    "unfolded",
    "original_shape",
    "navigation_mask",
    "signal_mask",
    "on_scores",
)


class LearningResults:
    """Stores the parameters and results from a decomposition.

    Standalone copy for the hyperspy-ml package with .npz save/load
    backward-compatible with the legacy HyperSpy format.
    """

    components = None
    scores = None
    explained_variance = None
    explained_variance_ratio = None
    number_significant_components = None
    decomposition_algorithm = None
    poissonian_noise_normalized = None
    output_dimension = None
    mean = None
    centre = None
    cluster_membership = None
    cluster_labels = None
    cluster_centers = None
    cluster_centers_estimated = None
    cluster_algorithm = None
    number_of_clusters = None
    estimated_number_of_clusters = None
    cluster_metric_data = None
    cluster_metric_index = None
    cluster_metric = None
    bss_algorithm = None
    unmixing_matrix = None
    bss_components = None
    bss_scores = None
    unfolded = None
    original_shape = None
    navigation_mask = None
    signal_mask = None
    on_scores = False

    @property
    def factors(self):
        self._warn_deprecated("factors", "components")
        return self.components

    @factors.setter
    def factors(self, value):
        self._warn_deprecated("factors", "components")
        self.components = value

    @property
    def loadings(self):
        self._warn_deprecated("loadings", "scores")
        return self.scores

    @loadings.setter
    def loadings(self, value):
        self._warn_deprecated("loadings", "scores")
        self.scores = value

    @property
    def bss_factors(self):
        self._warn_deprecated("bss_factors", "bss_components")
        return self.bss_components

    @bss_factors.setter
    def bss_factors(self, value):
        self._warn_deprecated("bss_factors", "bss_components")
        self.bss_components = value

    @property
    def bss_loadings(self):
        self._warn_deprecated("bss_loadings", "bss_scores")
        return self.bss_scores

    @bss_loadings.setter
    def bss_loadings(self, value):
        self._warn_deprecated("bss_loadings", "bss_scores")
        self.bss_scores = value

    @property
    def on_loadings(self):
        self._warn_deprecated("on_loadings", "on_scores")
        return self.on_scores

    @on_loadings.setter
    def on_loadings(self, value):
        self._warn_deprecated("on_loadings", "on_scores")
        self.on_scores = value

    @staticmethod
    def _warn_deprecated(old_name, new_name):
        warnings.warn(
            f"`{old_name}` is deprecated and will be removed in HyperSpy "
            f"v3.0. Use `{new_name}` instead.",
            VisibleDeprecationWarning,
        )

    def save(self, filename, overwrite=None):
        """Save results to a .npz file."""
        kwargs = {}
        for name in _SAVEABLE_ATTRS:
            if name in self.__dict__:
                kwargs[name] = self.__dict__[name]
        if overwrite is None:
            overwrite = rs_path.overwrite(filename)
        if overwrite:
            np.savez(filename, **kwargs)
            _logger.info(f"Saved results to {filename}")

    def load(self, filename):
        """Load results from a .npz file with backward-compatible key migration."""
        decomposition = np.load(filename, allow_pickle=True)
        _d = self.__dict__
        _d.clear()
        for key, value in decomposition.items():
            if value.dtype == np.dtype("object"):
                value = None
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            _d[key] = value

        # Compatibility migrations
        _migrations = {
            "algorithm": "decomposition_algorithm",
            "V": "explained_variance",
            "w": "unmixing_matrix",
            "pca_algorithm": "decomposition_algorithm",
            "ica_algorithm": "bss_algorithm",
            "v": "scores",
            "pc": "scores",
            "W": "unmixing_matrix",
            "ica_scores": "bss_scores",
            "ica_factors": "bss_components",
            "factors": "components",
            "loadings": "scores",
            "bss_factors": "bss_components",
            "bss_loadings": "bss_scores",
            "on_loadings": "on_scores",
        }
        _remove_keys = {"variance2one", "centered"}
        for old_key, new_key in _migrations.items():
            if old_key in _d:
                _d[new_key] = _d[old_key]
                del _d[old_key]
        for key in _remove_keys:
            _d.pop(key, None)

        _logger.info(f"Loaded results from {filename}")
        return self

    def summary(self):
        text = (
            "Decomposition parameters\n"
            "------------------------\n"
            f"normalize_poissonian_noise={self.poissonian_noise_normalized}\n"
            f"algorithm={self.decomposition_algorithm}\n"
            f"output_dimension={self.output_dimension}\n"
            f"centre={self.centre}"
        )
        if self.bss_algorithm is not None:
            text += (
                "\n\nDemixing parameters\n"
                "-------------------\n"
                f"algorithm={self.bss_algorithm}\n"
                f"n_components={len(self.unmixing_matrix)}"
            )
        _logger.info(text)
        return text

    def __repr__(self):
        return self.summary()

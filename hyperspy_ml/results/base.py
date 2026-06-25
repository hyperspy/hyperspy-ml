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

"""Minimal typed result containers for ML pipeline stages.

Task 7a will expand these with events, provenance tracking,
and richer metadata.  For now they are simple dataclasses that
the stage classes produce and consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    """

    components: np.ndarray | None = None
    scores: np.ndarray | None = None
    explained_variance: np.ndarray | None = None
    explained_variance_ratio: np.ndarray | None = None
    mean: np.ndarray | None = None
    bH: np.ndarray | None = None
    aG: np.ndarray | None = None
    n_components: int = 0
    centre: str | None = None
    algorithm: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

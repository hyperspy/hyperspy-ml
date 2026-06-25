"""HyperSpy ML — machine learning extension for HyperSpy.

Provides multivariate analysis tools (decomposition, BSS, clustering)
as a HyperSpy extension package.
"""

from importlib.metadata import version

from hyperspy_ml.api import bss, cluster, decompose, extract_results, load_result
from hyperspy_ml.results.base import BSSResult, ClusterResult, DecompositionResult
from hyperspy_ml.stages.bss import BSS
from hyperspy_ml.stages.clustering import Clustering
from hyperspy_ml.stages.decomposition import Decomposition

__version__ = version("hyperspy-ml")

__all__ = [
    "Decomposition",
    "BSS",
    "Clustering",
    "DecompositionResult",
    "BSSResult",
    "ClusterResult",
    "decompose",
    "bss",
    "cluster",
    "load_result",
    "extract_results",
]

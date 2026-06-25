# -*- coding: utf-8 -*-
"""Pytest configuration for hyperspy-ml test suite.

Skips legacy and algorithm-specific test files that use the old
signal-based API or belong to the hyperspy-ml-algorithms package.

New stage-based tests covering equivalent functionality:
  - test_decomposition_stage.py (30 tests)
  - test_incremental.py (24 tests)
  - test_model_reconstruction.py (25 tests)
  - test_scree_plot.py (25 tests)
  - test_bss_stage.py (10 tests)
  - test_clustering_stage.py (11 tests)
  - test_lazy_decomposition_stage.py (11 tests)
"""

collect_ignore = [
    # Legacy file — old signal.decomposition() API (covered by test_decomposition_stage.py)
    "test_decomposition.py",
    # Legacy file — old signal.blind_source_separation() API (covered by test_bss_stage.py)
    "test_bss.py",
    # Legacy file — old signal.cluster_analysis() API (covered by test_clustering_stage.py)
    "test_cluster.py",
    # Legacy file — old signal.learning_results export (not yet implemented in new API)
    "test_export.py",
    # Legacy file — old LearningResults repr (not applicable to new result types)
    "test_learning_results.py",
    # Legacy file — deprecated factors/loadings property aliases (not applicable to new API)
    "test_deprecation_factors_loadings.py",
    # Algorithm-specific tests (covered by hyperspy-ml-algorithms)
    "test_mlpca.py",
    "test_ornmf.py",
    "test_rpca.py",
    "test_svd_pca.py",
    "test_whitening.py",
    # Import-level failure (ISVD not available in standalone install)
    "test_incremental_svd.py",
    # Old test_lazy_decomposition.py classes: imported separately, skip old classes
    "test_lazy_decomposition.py",
]

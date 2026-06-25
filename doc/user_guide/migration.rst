.. _migration_guide:

Migration Guide
===============

This guide helps seasoned HyperSpy users transition to the new hyperspy-ml API.
The main change is the move from signal-attached methods (``signal.decomposition()``)
to a functional and result-oriented API (``decompose(signal) -> result``).

API Mapping Table
-----------------

+------------------------------------------+-------------------------------------------------------+
| Old HyperSpy API                         | New hyperspy-ml API                                   |
+==========================================+=======================================================+
| ``s.decomposition()``                    | ``decompose(s)``                                      |
+------------------------------------------+-------------------------------------------------------+
| ``s.blind_source_separation()``          | ``bss(result)``                                       |
+------------------------------------------+-------------------------------------------------------+
| ``s.cluster_analysis()``                 | ``cluster(result)``                                   |
+------------------------------------------+-------------------------------------------------------+
| ``s.learning_results``                   | ``DecompositionResult`` / ``BSSResult``               |
+------------------------------------------+-------------------------------------------------------+
| ``s.learning_results.save()``            | ``result.save()`` (new ``.hsml`` format)              |
+------------------------------------------+-------------------------------------------------------+
| ``s.get_decomposition_model()``          | ``result.get_decomposition_model()``                  |
+------------------------------------------+-------------------------------------------------------+
| ``s.plot_scree_plot()``                  | ``result.plot_scree()``                               |
+------------------------------------------+-------------------------------------------------------+
| ``s.plot_decomposition_components()``    | ``result.plot_components()``                          |
+------------------------------------------+-------------------------------------------------------+
| ``s.plot_decomposition_scores()``        | ``result.plot_scores()``                              |
+------------------------------------------+-------------------------------------------------------+
| ``s.plot_bss_components()``              | ``result.plot_bss_components()``                      |
+------------------------------------------+-------------------------------------------------------+

Key Concepts
------------

Result Objects
~~~~~~~~~~~~~~

In the new API, all machine learning results are returned as dedicated result
objects (``DecompositionResult``, ``BSSResult``, ``ClusterResult``). These
objects are decoupled from the signal, allowing you to store multiple results
for the same signal and manage them independently.

Saving and Loading
~~~~~~~~~~~~~~~~~~

The legacy ``.npz`` format is replaced by a new Zarr-based ``.hsml`` format.
This format is more robust and supports lazy loading of large results.

.. code-block:: python

   # Save
   result.save("my_analysis.hsml")

   # Load
   from hyperspy_ml import load_result
   result = load_result("my_analysis.hsml")

Legacy Compatibility
~~~~~~~~~~~~~~~~~~~~

You can extract results from old ``.hspy`` files or legacy ``.npz`` files using
the ``extract_results()`` function:

.. code-block:: python

   from hyperspy_ml import extract_results
   s = hs.load("old_file.hspy")
   result = extract_results(s)

New Features
------------

Study for Trial-and-Error
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Study`` container allows you to collect multiple results and compare them.
It is particularly useful for parameter sweeps or comparing different algorithms.

.. code-block:: python

   from hyperspy_ml.study import Study
   study = Study("My PCA Study")
   study.add(decompose(s, algorithm="SVD"), name="SVD_default")
   study.add(decompose(s, algorithm="NMF"), name="NMF_3comp")

Pipeline
~~~~~~~~

The ``Pipeline`` class allows you to chain multiple ML stages together into a
single executable workflow.

.. code-block:: python

   from hyperspy_ml.pipeline import Pipeline
   from hyperspy_ml.stages.decomposition import Decomposition
   from hyperspy_ml.stages.bss import BSS

   pipe = Pipeline([
       ("pca", Decomposition(algorithm="SVD", output_dimension=5)),
       ("bss", BSS(algorithm="orthomax"))
   ])
   bss_result = pipe.run(s)

.. _mva.export:

Export results
==============

Obtain the results as arrays
----------------------------

The decomposition and BSS results are stored as numpy arrays in the result objects.
You can obtain them using the following methods:

* :meth:`~hyperspy_ml.results.base.DecompositionResult.get_decomposition_scores`
* :meth:`~hyperspy_ml.results.base.DecompositionResult.get_decomposition_components`
* :meth:`~hyperspy_ml.results.base.DecompositionResult.get_bss_scores`
* :meth:`~hyperspy_ml.results.base.DecompositionResult.get_bss_components`

.. _mva.saving-label:

Save and load results
---------------------

hyperspy-ml introduces a new Zarr-based format (``.hsml``) for saving and loading
machine learning results independently of the signal data.

Save to a .hsml file
~~~~~~~~~~~~~~~~~~~~

You can save any result object (Decomposition, BSS, or Clustering) to a ``.hsml``
file using the :meth:`~hyperspy_ml.results.base.DecompositionResult.save` method:

.. code-block:: python

   >>> # Save the result of the analysis
   >>> result.save('my_results.hsml') # doctest: +SKIP

Load from a .hsml file
~~~~~~~~~~~~~~~~~~~~~~

To load a result back, use the :func:`~hyperspy_ml.load_result` function:

.. code-block:: python

   >>> from hyperspy_ml import load_result
   >>> result = load_result('my_results.hsml') # doctest: +SKIP

Extract results from legacy files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have old HyperSpy files (``.hspy``) that contain machine learning results
in the legacy ``learning_results`` attribute, or legacy ``.npz`` result files,
you can extract them into the new result objects using :func:`~hyperspy_ml.extract_results`:

.. code-block:: python

   >>> from hyperspy_ml import extract_results
   >>> # Extract from a signal loaded from a .hspy file
   >>> s = hs.load("legacy_file.hspy")
   >>> result = extract_results(s)

   >>> # Extract directly from a legacy .npz file
   >>> result = extract_results("legacy_results.npz")

Export in different formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can still export the results to any format supported by RosettaSciIO by
first converting the result arrays to HyperSpy signals:

.. code-block:: python

   >>> # Example: export components to a TIFF file
   >>> components_signal = hs.signals.Signal1D(result.components)
   >>> components_signal.save("components.tif") # doctest: +SKIP

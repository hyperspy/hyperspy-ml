.. _mva.visualization:

Visualizing results
===================

hyperspy-ml provides dedicated plotting methods on the result objects for
visualizing decomposition, blind source separation, and clustering results.

.. _mva.scree_plot:

Scree plots
-----------

.. note::
   Scree plots are only available for the ``"SVD"`` and ``"PCA"`` algorithms.

PCA will sort the components in the dataset in order of decreasing
variance. It is often useful to estimate the dimensionality of the data by
plotting the explained variance against the component index. This plot is
sometimes called a scree plot.

To obtain a scree plot for your dataset, run the
:meth:`~hyperspy_ml.results.base.DecompositionResult.plot_scree` method on the
decomposition result:

.. code-block:: python

   >>> result.plot_scree(n=20) # doctest: +SKIP

The point at which the scree plot becomes linear (often referred to as
the "elbow") is generally judged to be a good estimation of the dimensionality
of the data.

By specifying a ``threshold`` value, a cutoff line will be drawn at the total variance
specified.

.. code-block:: python

   >>> result.plot_scree(n=20, threshold=4, xaxis_type='number') # doctest: +SKIP

The number of significant components can be estimated and a vertical line
drawn to represent this by specifying ``vline=True``.

.. _mva.plot_decomposition:

Decomposition plots
-------------------

To plot the decomposition components and scores, use the following methods on
the :class:`~hyperspy_ml.results.base.DecompositionResult` object:

* :meth:`~hyperspy_ml.results.base.DecompositionResult.plot_components`
* :meth:`~hyperspy_ml.results.base.DecompositionResult.plot_scores`

.. code-block:: python

   >>> result.plot_components() # doctest: +SKIP
   >>> result.plot_scores() # doctest: +SKIP

.. _mva.plot_bss:

Blind source separation plots
-----------------------------

Visualizing blind source separation results is similar to decomposition.
Use the following methods on the :class:`~hyperspy_ml.results.base.DecompositionResult`
(after BSS results have been assigned) or :class:`~hyperspy_ml.results.base.BSSResult` object:

* :meth:`~hyperspy_ml.results.base.DecompositionResult.plot_bss_components`
* :meth:`~hyperspy_ml.results.base.DecompositionResult.plot_bss_scores`

.. _mva.plot_clustering:

Clustering plots
----------------

To visualize clustering results, use the following methods on the
:class:`~hyperspy_ml.results.base.ClusterResult` object:

* :meth:`~hyperspy_ml.results.base.ClusterResult.plot_cluster_signals`
* :meth:`~hyperspy_ml.results.base.ClusterResult.plot_cluster_labels`
* :meth:`~hyperspy_ml.results.base.ClusterResult.plot_cluster_distances`

.. code-block:: python

   >>> cluster_result.plot_cluster_signals() # doctest: +SKIP
   >>> cluster_result.plot_cluster_labels() # doctest: +SKIP

Welcome to convst documentation !
=================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   api
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Information

   reproducibility
   citation


**convst** is a Python package dedicated to the Convolutional Shapelet Transform (CST).

Minimal example
---------------

The following code snippet illustrates the basic usage of convst:

.. code-block:: python

    from convst.classifiers import R_DST_Ridge
    from convst.utils.dataset_utils import load_sktime_dataset_split

    X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split(
        'GunPoint', normalize=True
    )

    # First run may be slow due to numba compilations on the first call. 
    # Run small dataset like GunPoint if this is the first time you call RDST on your system.
    # You can change n_shapelets to 1 to make this process faster.

    rdst = R_DST_Ridge(n_shapelets=10_000).fit(X_train, y_train)

    print("Accuracy Score for RDST : {}".format(rdst.score(X_test, y_test)))


1. First we import the ``R_DST_Ridge`` class that containt a wrapper for R_DST using a Ridge classifier. We also import a data loading function

2. Then we load the training and test sets by calling the ``load_sktime_dataset_split`` function.

3. Finally we fit the model on the training set and evaluate its
   performance by computing the accuracy on the test set using the score function.

We try ou best to follow the guidelines of sklearn to ensure compatibility with 
their numerous tools. For more information visit the
`Scikit-learn compatibility <scikit_learn_compatibility.html>`_ page.


`Getting started <install.html>`_
---------------------------------

Information to install, test, and contribute to the package.

`User Guide <user_guide.html>`_
-------------------------------

The main documentation. This contains an in-depth description of all
algorithms and how to apply them.

`API Documentation <api.html>`_
-------------------------------

The exact API of all functions and classes, as given in the
docstrings. The API documents expected types and allowed features for
all functions, and all parameters available for the algorithms.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples illustrating the use of the different algorithms. It
complements the `User Guide <user_guide.html>`_.

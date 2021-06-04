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

    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import accuracy_score
    from convst.transformers import ConvolutionalShapeletTransformer
    from convst.utils import load_sktime_dataset_split
    
    # Load Dataset by name. Any name of the univariate UCR archive can work.
    X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split(
        'GunPoint', normalize=True)
    
    # First run will be slow due to numba compilations on the first call. Run small dataset like GunPoint the first time !
    # Put verbose = 1 to see the progression of the algorithm.
    
    cst = make_pipeline(
        ConvolutionalShapeletTransformer(verbose=0),
        RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                          normalize=True, class_weight='balanced')
    )
    
    cst.fit(X_train, y_train)
    pred = cst.predict(X_test)
    
    print("Accuracy Score for CST : {}".format(accuracy_score(y_test, pred)))

1. First we import sklearn components to build and evaluate a pipeline, CST, and a data loading function

2. Then we load the training and test sets by calling the ``load_sktime_dataset_split`` function.

3. Next we define a pipeline using CST and a Ridge classifier.

4. Finally we fit the pipeline on the training set and evaluate its
   performance by computing the accuracy on the test set.

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

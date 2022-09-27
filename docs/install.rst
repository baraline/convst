.. _install:

=====================================
Installation, testing and development
=====================================

Dependencies
------------

To be fully able to run the convst packages and the examples, the following packages are required:

    - matplotlib >= 3.5,
    - numba >= 0.55,
    - pandas >= 1.3,
    - scikit_learn >= 1.0,
    - joblib >= 1.1.0,
    - pyts >= 0.12,
    - scipy >= 1.7,
    - seaborn >= 0.11,
    - sktime >= 0.10,
    - numpy < 1.22, >=1.18,
    - networkx >= 2.6.3


User installation
-----------------

If you already have a working installation of numpy, scipy, scikit-learn,
joblib and numba, you can easily install convst using ``pip``::

    pip install convst

You can also get the latest version of convst by cloning the repository::

    git clone https://github.com/baraline/convst.git
    cd convst
    python setup.py install


Testing
-------

After installation, you can launch the test suite from outside the source
directory using ``pytest``::

    pytest convst

This is WIP


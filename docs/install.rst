.. _install:

=====================================
Installation, testing and development
=====================================

Dependencies
------------

To be fully able to run the convst packages and the examples, the following packages are required:

    - Python (>= 3.7)
    - NumPy (>= 1.18.5)
    - Pandas (>= 1.1.4)
    - SciPy (>= 1.5.0)
    - Scikit-Learn (>=0.24.0)
    - Numba (>=0.50.1)
    - Seaborn (>=0.11.1)
    - Matplotlib (>=3.2.2)
    - Sktime (>= 0.5.3)


User installation
-----------------

If you already have a working installation of numpy, scipy, scikit-learn,
joblib and numba, you can easily install pyts using ``pip``::

    pip install convst

You can also get the latest version of pyts by cloning the repository::

    git clone https://github.com/baraline/convst.git
    cd convst
    python setup.py install


Testing
-------

After installation, you can launch the test suite from outside the source
directory using ``pytest``::

    pytest convst

This is WIP


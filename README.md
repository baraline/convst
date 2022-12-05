
Welcome to the convst repository. It contains the implementation of the `Random Dilated Shapelet Transform (RDST)` along with other works in the same area.
This work was supported by the following organisations:

<p float="center">
  <img src="https://raw.githubusercontent.com/baraline/convst/main/docs/_static/img/logo-UO-2022.png" width="32%" />
  <img src="https://raw.githubusercontent.com/baraline/convst/main/docs/_static/img/logo-lifo.png" width="32%" /> 
  <img src="https://raw.githubusercontent.com/baraline/convst/main/docs/_static/img/Logo_Worldline_-_2021(1).png" width="32%" />
</p>

## Status

| Overview | |
|---|---|
| **Compatibility** | [![!python-versions](https://img.shields.io/pypi/pyversions/convst)](https://www.python.org/)
| **CI/CD** |  [![!pypi](https://img.shields.io/pypi/v/convst?color=orange)](https://pypi.org/project/convst/)  ![docs](https://img.shields.io/readthedocs/convst) ![build](https://github.com/baraline/convst/actions/workflows/test.yml/badge.svg)| 
| **Code Quality** |  [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/baraline/convst.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/baraline/convst/context:python) [![Total alerts](https://img.shields.io/lgtm/alerts/g/baraline/convst.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/baraline/convst/alerts/) ![lines](https://img.shields.io/tokei/lines/github/baraline/convst) [![CodeFactor](https://www.codefactor.io/repository/github/baraline/convst/badge/main)](https://www.codefactor.io/repository/github/baraline/convst/overview/main) |
| **Downloads**| [![Downloads](https://pepy.tech/badge/convst)](https://pepy.tech/project/convst) |



<p float="center">
  <img src="https://raw.githubusercontent.com/baraline/convst/main/docs/_static/img/cd_ensemble.png" width="100%" />
</p>

## Installation

The recommended way to install the latest stable version is to use pip with `pip install convst`. To install the package from sources, you can download the latest version on GitHub and run `python setup.py install`. This should install the package and automatically look for the dependencies using `pip`. 

We recommend doing this in a new virtual environment using anaconda to avoid any conflict with an existing installation. If you wish to install dependencies individually, you can see dependencies in the `setup.py` file.

An optional dependency that can help speed up numba, which is used in our implementation, is the Intel vector math library (SVML). When using conda it can be installed by running `conda install -c numba icc_rt`. I didn't test the behavior with AMD processors, but I suspect it won't work.

## Tutorial
We give here a minimal example to run the `RDST` algorithm on any dataset of the UCR archive using the sktime API to fect dataset:

```python

from convst.classifiers import R_DST_Ridge
from convst.utils.dataset_utils import load_sktime_dataset_split

X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split('GunPoint')

# First run may be slow due to numba compilations on the first call. 
# Run a small dataset like GunPoint if this is the first time you call RDST on your system.
# You can change n_shapelets to 1 to make this process faster. The n_jobs parameter can
# also be changed to increase speed once numba compilation are done.

rdst = R_DST_Ridge(n_shapelets=10_000, n_jobs=1).fit(X_train, y_train)

print("Accuracy Score for RDST : {}".format(rdst.score(X_test, y_test)))
```
You can also visualize a shapelet using the visualization tool to obtain such visualization :

![Example of shapelet visualization](https://raw.githubusercontent.com/baraline/convst/main/docs/_static/img/shp_vis.png)

To know more about all the interpretability tools, check the documentation on readthedocs.

## Supported inputs
RDST support the following type of time series:
- Univariate and same length
- Univariate and variable length
- Multivariate and same length
- Multivariate and variable length

We use the standard scikit-learn interface and expect as input a 3D numpy array of shape `(n_samples, n_features, n_timestamps)`. For variable length input, we expect a (python) list of numpy arrays, or a numpy array with object dtype.

## Reproducing the paper results

Multiple scripts are available under the `PaperScripts` folder. It contains the exact same scripts used to generate our results, notably the `test_models.py` file, used to generate the csv results available in the `Results` folder of the archive (To be updated).

## Contributing, Citing and Contact

If you are experiencing bugs in the RDST implementation, or would like to contribute in any way, please create an issue or pull request in this repository.
For other question or to take contact with me, you can email me at antoine.guillaume45@gmail.com

If you use our algorithm or publication in any work, please cite the following paper (ArXiv version https://arxiv.org/abs/2109.13514):

```bibtex
@InProceedings{10.1007/978-3-031-09037-0_53,
author="Guillaume, Antoine
and Vrain, Christel
and Elloumi, Wael",
title="Random Dilated Shapelet Transform: A New Approach for Time Series Shapelets",
booktitle="Pattern Recognition and Artificial Intelligence",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="653--664",
abstract="Shapelet-based algorithms are widely used for time series classification because of their ease of interpretation, but they are currently outperformed by recent state-of-the-art approaches. We present a new formulation of time series shapelets including the notion of dilation, and we introduce a new shapelet feature to enhance their discriminative power for classification. Experiments performed on 112 datasets show that our method improves on the state-of-the-art shapelet algorithm, and achieves comparable accuracy to recent state-of-the-art approaches, without sacrificing neither scalability, nor interpretability.",
isbn="978-3-031-09037-0"
}


```

## TODO for relase 1.0:

- [ ] Finish Numpy docs in all python files
- [ ] Update documentation and examples
- [X] Enhance interface for interpretability tools
- [X] Add the Generalised version of RDST
- [ ] Continue unit tests and code coverage/quality
- [ ] Add function signatures to numba functions

## Citations

Here are the code-related citations that were not made in the paper

[1]: [The Scikit-learn development team, "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research 2011](https://scikit-learn.org/stable/)

[2]: [The Numpy development team, "Array programming with NumPy", Nature 2020](https://numpy.org/)

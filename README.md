[![Documentation Status](https://readthedocs.org/projects/convst/badge/?version=latest)](https://convst.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/convst)](https://pepy.tech/project/convst)

17/11/21 : Work on this project will now resume. The new paper and code version are expected to be live in Febuary 2022.

This repository contains the implementation of the `Convolutional Shapelet Transform (CST)`, a state-of-the-art shapelet algorithm.
It compute a set of convolutional shapelets that match small parts of the input space with highly discriminative points in multiple convolutional spaces.

!! The paper will be reworked before publication to better explain the algorithm and results, and include our latest improvements. Change to the code are to be expected !!

## Installation

The package support Python 3.7 & 3.8 (3.9 untested).  You can install the package and its dependencies via pip using `pip install convst`. To install the package from sources you can download the latest release on github and run `python setup.py install`. This will install the package and automaticaly look for the dependencies using `pip`. 

We recommend doing this in a new virtual environment using anaconda to avoid any conflict with an existing installation. If you wish to install dependencies individually, you can the strict dependencies used in the `requierements.txt` file.

An optional dependency that can help speed up numba, which is used in our implementation is the Intel vector math library (SVML). When using conda it can be installed by running `conda install -c numba icc_rt`

Requiered packages do not include packages not related to CST, the following packages could be useful if you want to run some other scripts in the archive:

0. `wildboard` used for ShapeletForestClassifier
1. `networkx` used to generate critical difference diagrams

## Tutorial
We give here a minimal example to run the `CST` algorithm on any univariate dataset of the UCR archive:

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from convst.transformers.convolutional_ST import ConvolutionalShapeletTransformer
from convst.utils import load_sktime_dataset_split

# Load Dataset by name. Any name of the univariate UCR archive can work.
X_train, X_test, y_train, y_test, _ = load_sktime_dataset_split(
    'GunPoint', normalize=True)

# First run may be slow due to numba compilations on the first call. 
# Run small dataset like GunPoint if this is the first time you call CST on your system.
# Put verbose = 1 to see the progression of the algorithm.

cst = make_pipeline(
    ConvolutionalShapeletTransformer(verbose=0),
    RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                      normalize=True, class_weight='balanced')
)

cst.fit(X_train, y_train)
pred = cst.predict(X_test)

print("Accuracy Score for CST : {}".format(accuracy_score(y_test, pred)))
```

We use the standard scikit-learn interface and expect as input a 3D matrix of shape `(n_samples, n_features, n_timestamps)`. Note that as only univariate is supported for now, CST will only process the first feature.

In the `Example` folder, you can find some other scripts to help you get started and show you how to plot some results. The `UCR_example.py` script allows you to run CST on any UCR dataset and plot interpretations of the results.
Additional experiments mentioned in the paper are also found in this folder.

## Current Work in Progress

The package currently has some limitations that are being worked on, the mains ones being:

- [ ] Adaptation to the multivariate context. While you can feed a multivariate time series to CST, it will only look at the first feature for now.
- [ ] Adaptation to unsupervised context. The ideal being to implement a clustering version of the algortihm using scikit-learn standards.
- [ ] Possibility to change the model used to extract partitions of the data in CST.
- [ ] Parallel implementation of the remaining sequential parts of CST and global optimizations to speed-up CST.
- [ ] Use of more diverse set of features extracted from the convolutions, notably those from Catch-22.
- [ ] Redisgn interpretability tool to be more resilient to context (supervised or not) and high number of "class", currently graphs are really messy with high number of classes.
- [X] ~~Special case testing show a potential issue when class difference can only be made by value at a particular timepoint (with noise), a fix is in progress.~~ Changes will be available in next release (0.1.5). Increase in performance on previously problematic problems while retaining same performance on the others and some improvments on runtime and memory consumption. 

The `benchmark` folder offer visualisations of the performance change between release versions where CST was modified.

## Reproducing the paper results

Multiple scripts are available under the `PaperScripts` folder. It contains the exact same scripts used to generate our results.

To obtain the same resampling data as the UCR archive, you muse use the [tsml](https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/examples/DataHandling.java) java repository, then from the class `DataHandling` in the package examples, use the function `resamplingData` and change where to read and write the data from. The function assumes the input is in arff format, they can be obtained on the [time serie classification website](http://www.timeseriesclassification.com/)

## Contributing, Citing and Contact

If you are experiencing bugs in the CST implementation, or would like to contribute in any way, please create an issue or pull request in this repository
For other question, you can create an issue on this repository.

If you use our algorithm or publication in any work, please cite the following paper https://arxiv.org/abs/2109.13514 :
```bibtex
@misc{guillaume2021convolutional,
  title={Convolutional Shapelet Transform: A new approach for time series shapelets},
  author={Guillaume Antoine, Vrain Christel, Elloumi Wael},
  eprint={2109.13514},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  year={2021}
  publisher={}
}
```
## Citations

Here are the code-related citations that were not made in the paper

[1]: [Loning, Markus and Bagnall, Anthony and Ganesh, Sajaysurya and Kazakov, Viktor and Lines, Jason and Kiraly, Franz J, "sktime: A Unified Interface for Machine Learning with Time Series", Workshop on Systems for ML at NeurIPS 2019}](https://www.sktime.org/en/latest/)


[2]: [The Scikit-learn development team, "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research 2011](https://scikit-learn.org/stable/)


[3]: [The Numpy development team, "Array programming with NumPy", Nature 2020](https://numpy.org/)


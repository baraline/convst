This repository contains the implementation of the `Convolutional Shapelet Transform (CST)`, a state-of-the-art shapelet algorithm.
It compute a set of convolutional shapelets that match small parts of the input space with highly discriminative points in multiple convolutional spaces.

## Installation

The repository was developped under Python 3.8. We will guarantee the support of Python 3.7 in future version, although the modifications might be minors or inexistant. 
To install the package and run the example you must :

0. If you are making a new installation, first install python, pip and setuptools.
1. Clone the repository https://github.com/baraline/CST.git
3. run `python3 setup.py install`

This will install the package and the dependencies. We do not yet support installation via `pip` in the initial release.
If you wish to install dependencies individually, you can the strict dependencies used in the `requierements.txt` file

## Tutorial
We give here a minimal example to run the `CST` algorithm on any dataset of the UCR/UEA archive:

```python
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# Load Dataset by name, here we use 'GunPoint'
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)
cst = ConvolutionalShapeletTransformer().fit(X_train, y_train)
X_cst_train = cst.transform(X_train)
X_cst_test = cst.transform(X_test)
rf =  RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),normalize=True).fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("F1-Score for CST RF : {}".format(f1_score(y_test, pred)))
```

We use the standard scikit-learn interface and expect as input a 3D matrix of shape `(n_samples, n_features, n_timestamps)`, altought we didn't yet extended the approach to the multivariate context, one can use the `id_ft` parameter of CST to change on which feature the algorithm is computing the transform.

In the `Example` folder, you can find some other scripts to help you get started and show you how to plot some results. Additional experiments mentioned in the paper are also found in this folder.

## Reproducing the paper results and figures

Multiple scripts are available under the `PaperScripts` folder. It contains the exact same scripts used to build figures and LaTeX tables we use in our paper.

To obtain the same resampling data as the UCR archive, you muse use the [tsml](https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/examples/DataHandling.java) java repository, then from the class `DataHandling` in the package examples, use the function `resamplingData` and change where to read and write the data from. 
The function assumes the input is in arff format, they can be obtained on the [time serie classification website](http://www.timeseriesclassification.com/)

## Contributing, Citing and Contact

If you are experiencing bugs in the CST implementation, or would like to contribute in any way, please create an issue or pull request in this repository
For other question or to take contact with me, you can email me at XXXX (institutional email might change soon so i provide this as a temporary address)

If you use our algorithm or publication in any work, please cite the following paper :
```bibtex
@article{CST,
  title={Convolutional Shapelet Transform: A new approach for time series shapelets},
  author={Guillaume Antoine, Vrain Christel, Elloumi Wael},
  journal={},
  volume={},
  number={},
  pages={},
  year={2021}
  publisher={}
}
```
## License
[GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Citations
Here are the code-related citations that were not made in the paper

[2]: [Loning, Markus and Bagnall, Anthony and Ganesh, Sajaysurya and Kazakov, Viktor and Lines, Jason and Kiraly, Franz J, "sktime: A Unified Interface for Machine Learning with Time Series", Workshop on Systems for ML at NeurIPS 2019}](https://www.sktime.org/en/latest/)


[3]: [The Scikit-learn development team, "Scikit-learn: Machine Learning in Python", Journal of Machine Learning Research 2011](https://scikit-learn.org/stable/)


[4]: [The Numpy development team, "Array programming with NumPy", Nature 2020](https://numpy.org/)


This repository contains the implementation of the `Convolutional Shapelet Transform (CST)`, an algorithm that search for a set of shapelets that match small parts of the input space with highly discriminative points in multiple convolutional spaces.

## Installation

The repository was developped under Python 3.8. We will extend support to Python 3.7 in future version, although the modifications might be minors. 
To install the package and run the example you must :

1. Clone the repository https://github.com/baraline/CST.git
2. run `(python3 -m) pip -r install requirements.txt`

This will install the package and the dependencies.

## Tutorial
We give here a minimal example to run the `CST` algorithm on any dataset of the UCR/UEA archive:

```python
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import f1_score
# Load Dataset by name, here we use 'GunPoint'
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'GunPoint', normalize=True)
cst = MiniConvolutionalShapeletTransformer().fit(X_train, y_train)
X_cst_train = cst.transform(X_train)
X_cst_test = cst.transform(X_test)
rf = RandomForestClassifier(n_estimators=400).fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("F1-Score for CST RF : {}".format(f1_score(y_test, pred)))
```

We expect as input a 3D matrix of shape `(n_samples, n_features, n_timestamps)`, altought we didn't yeat extend the approach to the multivariate context, one can use the `id_ft` parameter of CST to change on which feature the algorithm is computing the transform

## Reproducing the paper results and figures

Multiple scripts are available under `/PaperScripts/`, it contains the scripts used to build figures and LaTeX tables we use in our paper.

## Contributing, Citing and Contact

If you are experiencing bugs in the CST implementation, or would like to contribute in any way, please create an issue or pull request in this repository
For other question or to take contact with me, you can email me at antoine(dot)guillaume45(at)gmail(dot)com (institutional email might change soon so i provide this as a temporary address)

If you use our algorithm in a publication, please cite the following paper :
```bibtex
@article{CST,
  title={Convolutional Shapelet Transform: A new approach for time series shapelets},
  author={Guillaume Antoine, Vrain Chrsitel, Elloumi Weal},
  journal={},
  volume={},
  number={},
  pages={},
  year={}
  publisher={}
}
```
# -*- coding: utf-8 -*-

__all__ = [
	"check_array_3D",
    "check_array_2D",
    "check_array_1D"
]

import numpy as np
import pandas as pd
from sktime.utils.data_processing import from_nested_to_3d_numpy, is_nested_dataframe

def check_array_3D(X, coerce_to_numpy=True, is_univariate=False):
    X = check_is_numpy_or_pd(X)
    if X.ndim != 3:
        raise ValueError(
            "If passed as a np.array, X must be a 3-dimensional "
            "array, but found shape: {}".format(X.shape)
        )
    if isinstance(X, pd.DataFrame):
        if not is_nested_dataframe(X):
            raise ValueError(
                "If passed as a pd.DataFrame, X must be a nested "
                "pd.DataFrame, with pd.Series or np.arrays inside cells."
            )
        if coerce_to_numpy:
            X = from_nested_to_3d_numpy(X)
    if is_univariate:
        if X.shape[1] != 1:
            raise ValueError(
                "X must be a 3-dimensional array with dimension 1 equal to 1"
            )
    return X

def check_array_2D(X, coerce_to_numpy=True):
    X = check_is_numpy_or_pd(X)
    if X.ndim != 2:
        raise ValueError(
            "If passed as a np.array, X must be a 2-dimensional "
            "array, but found shape: {}".format(X.shape)
        )
    if isinstance(X, pd.DataFrame):
        if coerce_to_numpy:
            X = X.values
    return X

def check_array_1D(X):
    X = check_is_numpy(X)
    if X.ndim != 1:
        raise ValueError(
            "If passed as a np.array, X must be a 1-dimensional "
            "array, but found shape: {}".format(X.shape)
        )
    
    return X

def check_is_numpy_or_pd(X):
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise ValueError(
            "Expected an pandas DataFrame or numpy array as input "
            "but got {}".format(str(type(X)))
        )

def check_is_numpy(X):
    if isinstance(X, list):
        return np.asarray(X)
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise ValueError(
            "Expected an python list or numpy array as input "
            "but got {}".format(str(type(X)))
        )


        
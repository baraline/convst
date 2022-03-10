# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy, is_nested_dataframe

def check_array_3D(X, coerce_to_numpy=True, is_univariate=False, min_timestamps=2):
    """
    Perform checks on the input to verify if it is a 3D array.

    Parameters
    ----------
    X : DataFrame or array, shape = (n_samples, n_features, n_timestamps)
        Input Time series
    coerce_to_numpy : boolean, optional
        If input is a pandas DataFrame, will convert it to numpy. The default is True.
    is_univariate : boolean, optional
        If true, will raise an error if X as more than 1 feature. The default is False.

    Raises
    ------
    ValueError
        

    Returns
    -------
    X : (n_samples, n_features, n_timestamps)
        Input Time series.

    """
    X = check_is_numpy_or_pd(X)
    if X.ndim != 3:
        raise ValueError(
            "If passed as a np.array, X must be a 3-dimensional "
            "array, but found shape: {}".format(X.shape)
        )
    if X.size == 0:
        raise ValueError(
            "Input is empty or have a dimension of size 0"
            ", found shape: {}".format(X.shape)
        )
    if X.shape[2] <= min_timestamps:
        raise ValueError(
            "Input should have more than {} timestamp"
            ", found only: {}".format(min_timestamps,X.shape[2])
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
    """
    Perform checks on the input to verify if it is a 2D array.

    Parameters
    ----------
    X : DataFrame or array, shape = (n_samples, n_timestamps)
        Input Time series
    coerce_to_numpy : boolean, optional
        If input is a pandas DataFrame, will convert it to numpy. The default is True.

    Raises
    ------
    ValueError
        

    Returns
    -------
    X : (n_samples, n_timestamps)
        Input Time series.

    """
    X = check_is_numpy_or_pd(X)
    if X.ndim != 2:
        raise ValueError(
            "If passed as a np.array, X must be a 2-dimensional "
            "array, but found shape: {}".format(X.shape)
        )
    if X.size == 0:
        raise ValueError(
            "Input is empty or have a dimension of size 0"
            ", found shape: {}".format(X.shape)
        )
    if isinstance(X, pd.DataFrame):
        if coerce_to_numpy:
            X = X.values
    return X

def check_array_1D(X):
    """
    Perform checks on the input to verify if it is a 1D array.

    Parameters
    ----------
    X : array, shape = (n_timestamps)
        Input Time series

    Raises
    ------
    ValueError
        

    Returns
    -------
    X : (n_timestamps)
        Input Time series.

    """
    X = check_is_numpy(X)
    if X.ndim != 1:
        raise ValueError(
            "If passed as a np.array, X must be a 1-dimensional "
            "array, but found shape: {}".format(X.shape)
        )
    if X.size == 0:
        raise ValueError(
            "Input is empty or have a dimension of size 0"
            ", found shape: {}".format(X.shape)
        )
    
    return X

def check_is_numpy_or_pd(X):
    """
    Check if the input is a numpy array or a pandas DataFrame, else raise
    an error

    Parameters
    ----------
    X : DataFrame or array
        Input data

    Raises
    ------
    ValueError

    Returns
    -------
    X : DataFrame or array
        Input data.

    """
    if isinstance(X, list):
        return np.asarray(X)
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise ValueError(
            "Expected an pandas DataFrame or numpy array or python list as input "
            "but got {}".format(str(type(X)))
        )

def check_is_numpy(X):
    """
    Check if the input is a numpy array, else raise an error

    Parameters
    ----------
    X : array
        Input data

    Raises
    ------
    ValueError

    Returns
    -------
    X : array
        Input data.

    """
    if isinstance(X, list):
        return np.asarray(X)
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise ValueError(
            "Expected an python list or numpy array as input "
            "but got {}".format(str(type(X)))
        )


        
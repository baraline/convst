# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:46:14 2021

@author: A694772
"""
__all__ = [
	"check_array_3D",
    "check_array_2D"
]

import numpy as np
import pandas as pd
from sktime.utils.data_processing import from_nested_to_3d_numpy, is_nested_dataframe

def check_array_3D(X, coerce_to_numpy=True):
    if X.ndim != 3:
        raise ValueError(
            "If passed as a np.array, X must be a 3-dimensional "
            "array, but found shape: {X.shape}"
        )
    if isinstance(X, pd.DataFrame):
        if not is_nested_dataframe(X):
            raise ValueError(
                "If passed as a pd.DataFrame, X must be a nested "
                "pd.DataFrame, with pd.Series or np.arrays inside cells."
            )
        # convert pd.DataFrame
        if coerce_to_numpy:
            X = from_nested_to_3d_numpy(X)
    return X

def check_array_2D(X, coerce_to_numpy=True):
    if X.ndim != 2:
        raise ValueError(
            "If passed as a np.array, X must be a 2-dimensional "
            "array, but found shape: {X.shape}"
        )
    if isinstance(X, pd.DataFrame):
        # convert pd.DataFrame
        if coerce_to_numpy:
            X = X.values
    return X
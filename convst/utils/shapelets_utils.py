# -*- coding: utf-8 -*-

from numpy.lib.stride_tricks import as_strided
from numba import njit

@njit(cache=True)
def generate_strides_1D(X, window_size, dilation):
    """
    Generate strides from the input univariate time series with specified 
    length and dilation parameters.

    Parameters
    ----------
    X : array, shape = (n_timestamps)
        An univariate time series, in a 1 dimensional view.
    window_size : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_samples, n_strides, stride_len)
        All possible subsequences of length stride_len for each time series.
    """
    n_timestamps = X.shape[0]
    shape_new = (n_timestamps - (window_size-1)*dilation,
                 window_size)

    s0 = X.strides[0]
    strides_new = (s0, dilation * s0)
    return as_strided(X, shape=shape_new, strides=strides_new)

@njit(cache=True)
def generate_strides_2D(X, window_size, dilation):
    """
    Generate strides from an ensemble of univariate time series with specified 
    length and dilation parameters.

    Parameters
    ----------
    X : array, shape = (n_samples, n_timestamps)
        An ensemble of univariate time series, in a 2 dimensional view.
    window_size : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_samples, n_strides, stride_len)
        All possible subsequences of length stride_len for each time series.
    """
    n_samples, n_timestamps = X.shape
    
    shape_new = (n_samples,
                 n_timestamps - (window_size-1)*dilation,
                 window_size)
    s0, s1 = X.strides
    strides_new = (s0, s1, dilation *s1)
    return as_strided(X, shape=shape_new, strides=strides_new) 
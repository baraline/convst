# -*- coding: utf-8 -*-

from numpy.lib.stride_tricks import as_strided
from numba import njit, prange

from convst.utils.numba_utils import *

###############################################################################
#                                                                             #
#                         DISTANCE FUNCTIONS                                  #
#                                                                             #
###############################################################################

@njit(
  ["float64(float64[:],float64[:])","float32(float32[:],float32[:])"],
  fastmath=True, cache=True
)
def euclidean(x, y):
    s = 0
    for i in prange(x.shape[0]):
        s += (x[i]-y[i])**2
    return np.sqrt(s)

@njit(
  ["float64(float64[:],float64[:])","float32(float32[:],float32[:])"],
  fastmath=True, cache=True
)
def squared_euclidean(x, y):
    s = 0
    for i in prange(x.shape[0]):
        s += (x[i]-y[i])**2
    return s

@njit(
  ["float64(float64[:],float64[:])","float32(float32[:],float32[:])"],
  fastmath=True, cache=True
)
def manhattan(x, y):
    s = 0
    for i in prange(x.shape[0]):
        s += abs(x[i]-y[i])
    return s

###############################################################################
#                                                                             #
#                    SUBSEQUENCE EXTRACTION FUNCTIONS                         #
#                                                                             #
###############################################################################


@njit(
  ["float64[:,:](float64[:], int64, int64)",
   "float32[,:,:](float32[:], int32, int32)",
   "int64[,:,:](int64[:], int64, int64)",
   "int32[:,:](int32[:], int32, int32)"],
  cache=True
)
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

@njit(
  ["float64[:,:,:](float64[:,:], int64, int64)",
   "float32[:,:,:](float32[:,:], int32, int32)",
   "int64[:,:,:](int64[:,:], int64, int64)",
   "int32[:,:,:](int32[:,:], int32, int32)"],
  cache=True
)
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

@njit(
  ["float64[:,:](float64[:], int64, int64)",
   "float32[,:,:](float32[:], int32, int32)",
   "int64[,:,:](int64[:], int64, int64)",
   "int32[:,:](int32[:], int32, int32)"],
  cache=True
)
def _generate_strides_1D_phase(X, l, d):
    n_timestamps = X.shape[0]
    X_new = np.zeros((n_timestamps,l))
    for i in prange(n_timestamps):
        for j in prange(l):
            X_new[i,j] = X[(i+(j*d))%n_timestamps]
    return X_new


@njit(
  ["float64[:,:,:](float64[:,:], int64, int64)",
   "float32[:,:,:](float32[:,:], int32, int32)",
   "int64[:,:,:](int64[:,:], int64, int64)",
   "int32[:,:,:](int32[:,:], int32, int32)"],
  cache=True
)
def _generate_strides_2D_phase(x, l, d):
    n_features, n_timestamps = x.shape
    X_new = np.zeros((n_features, n_timestamps, l))
    for ft in prange(n_features):
        for i in prange(n_timestamps):
            for j in prange(l):
                X_new[ft,i,j] = x[ft,(i+(j*d))%n_timestamps]
    return X_new
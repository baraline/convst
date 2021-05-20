# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:46:14 2021

@author: A694772
"""
__all__ = [
	"compute_distances",
    "min_dist_shp",
    "generate_strides_2D",
    "generate_strides_1D",
    "shapelet_dist_numpy"
]

import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_distances(X_strides, subsequences):
    """
    Compute the minimum euclidean distances between an ensemble of time series
    and an ensemble of subsequences.

    Parameters
    ----------
    X_strides : array, shape = (n_samples, n_strides, stride_len)
        All possible subsequences of length stride_len for each time series.
        
    subsequences : array, shape = (n_subsequences, stride_len)
        Subsequences on which we want to compute the minimal euclidean distances

    Returns
    -------
    dist_to_X : array, shape = (n_samples, n_subsequences)
        The minimal euclidean distance between each sample and each subsequences

    """
    dist_to_X = np.zeros((X_strides.shape[0], subsequences.shape[0]))
    for i in prange(X_strides.shape[0]):
        for j in prange(subsequences.shape[0]):
            dist_to_X[i,j] = min_dist_shp(X_strides[i], subsequences[j])
    return dist_to_X

@njit(fastmath=True)
def min_dist_shp(x_strides, subseq):
    """
    Compute the minimum euclidean distance from strides of a times series to a 
    subsequence of the length of the input strides

    Parameters
    ----------
    x_strides : array, shape = (n_strides, stride_len)
        All possible subsequences of length stride_len of a time series.

    subseq : array, shape = (stride_len)
        Subsequence on which we want to compute the minimal euclidean distance

    Returns
    -------
    float
        The minimum euclidean distance between the input strides and 
        the subsequence

    """
    d = np.zeros(x_strides.shape[0])
    for i in prange(x_strides.shape[0]):
        d[i] = np.linalg.norm(x_strides[i]-subseq) 
    return np.min(d)


def euclidean_pure_numpy(x, y):
    """Euclidean square distance matrix.
    
    Inputs:
    x: (N,) numpy array
    y: (N,) numpy array
    
    Ouput:
    (N, N) Euclidean square distance matrix:
    r_ij = x_ij^2 - y_ij^2
    """

    x2 = np.einsum('ij,ij->i', x, x)[:, np.newaxis]
    y2 = np.einsum('ij,ij->i', y, y)[:, np.newaxis].T

    xy = np.dot(x, y.T)

    return np.abs(x2 + y2 - 2. * xy)


def shapelet_dist_numpy(X_strides, subsequences):
    d = np.zeros((X_strides.shape[0],subsequences.shape[0]))
    for i in range(X_strides.shape[0]):
        d[i,:] += np.min(euclidean_pure_numpy(X_strides[i],subsequences),axis=0)
    return d
    

def generate_strides_2D(X, window, dilation):
    """
    Generate strides from an ensemble of univariate time series with specified 
    length and dilation parameters.

    Parameters
    ----------
    X : array, shape = (n_samples, n_timestamps)
        An ensemble of univariate time series, in a 2 dimensional view.
    window : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_samples, n_strides, stride_len)
        All possible subsequences of length stride_len for each time series.

    """
    n_rows, n_columns = X.shape
    shape = (n_rows, n_columns - ((window-1)*dilation), window)
    strides = np.array([X.strides[0], X.strides[1], X.strides[1]*dilation])
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)  

def generate_strides_1D(ts, window, dilation):
    """
    Generate strides from the input univariate time series with specified 
    length and dilation parameters.

    Parameters
    ----------
    ts : array, shape = (n_timestamps)
        An univariate time series, in a 1 dimensional view.
    window : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_samples, n_strides, stride_len)
        All possible subsequences of length stride_len for each time series.

    """
    shape = (ts.size - ((window-1)*dilation), window)
    strides = np.array([ts.strides[0], ts.strides[0]*dilation])
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)  

@njit(fastmath=True)
def min_dist_shp_loc(x_strides, subseq):
    """
    Compute the minimum euclidean distance from strides of a times series to a 
    subsequence of the length of the input strides

    Parameters
    ----------
    x_strides : array, shape = (n_strides, stride_len)
        All possible subsequences of length stride_len of a time series.

    subseq : array, shape = (stride_len)
        Subsequence on which we want to compute the minimal euclidean distance

    Returns
    -------
    (float, int)
        The minimum euclidean distance between the input strides and 
        the subsequence and the first location of this minimum

    """
    d = np.array([np.sum(np.power((x_strides[i]-subseq),2))
                  for i in prange(x_strides.shape[0])])
    return np.min(d), np.argmin(d)

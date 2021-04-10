# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:46:14 2021

@author: A694772
"""
__all__ = [
	"apply_one_kernel_one_sample",
    "apply_one_kernel_all_sample"
]
from numba import njit, prange
import numpy as np

@njit(fastmath=True)
def apply_one_kernel_one_sample(x, n_timestamps, weights, length, bias, 
                                dilation, padding):
    """
    Apply one kernel to one time series.

    Parameters
    ----------
    x : array, shape = (n_timestamps,)
        One time series.

    n_timestamps : int
        Number of timestamps.

    weights : array, shape = (length,)
        Weights of the kernel. Zero padding values are added.

    length : int
        Length of the kernel.

    bias : int
        Bias of the kernel.

    dilation : int
        Dilation of the kernel.

    padding : int
        Padding of the kernel.

    Returns
    -------
    x_new : array, shape = (2,)
        Extracted features using the kernel.

    """
    n_conv = n_timestamps - ((length - 1) * dilation) + (2 * padding)
    # Compute padded x
    if padding > 0:
        x_pad = np.zeros(n_timestamps + 2 * padding)
        x_pad[padding:-padding] = x
    else:
        x_pad = x

    # Compute the convolutions
    x_conv = np.zeros(n_conv)
    for i in prange(n_conv):
        for j in prange(length):
            x_conv[i] += weights[j] * x_pad[i + (j * dilation)]
    x_conv += bias
    return x_conv  

@njit(parallel=True)
def apply_one_kernel_all_sample(X, id_ft, weights, length, bias,
                                dilation, padding):
    n_samples, _, n_timestamps = X.shape
    n_conv = n_timestamps - ((length - 1) * dilation) + (2 * padding)
    X_conv = np.zeros((n_samples, 1, n_conv))
    for i in prange(n_samples):
        X_conv[i,0,:] = apply_one_kernel_one_sample(X[i,id_ft], n_timestamps,
                                                    weights, length, bias, 
                                                    dilation, padding)
    return X_conv


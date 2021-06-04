# -*- coding: utf-8 -*-

from numba import njit, prange
import numpy as np

@njit(fastmath=True)
def apply_one_kernel_one_sample(x, n_timestamps, weights, length, bias, 
                                dilation, padding):
    """
    Apply the convolution operation given the parameters of a convolutional kernel
    and one time series.

    Parameters
    ----------
    x : array, shape = (n_timestamps)
        One time series.
    n_timestamps : int
        Number of timestamps.
    weights : array, shape = (length)
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
    x_conv : array, shape = (n_convolution)
        The convovled input time series x.

    """
    n_conv = n_timestamps - ((length - 1) * dilation) + (2 * padding)
    # Compute padded x
    if padding > 0:
        x_pad = np.zeros(n_timestamps + 2 * padding,dtype=np.float32)
        x_pad[padding:-padding] = x
    else:
        x_pad = x

    # Compute the convolutions
    x_conv = np.zeros(n_conv,dtype=np.float32)
    for i in prange(n_conv):
        for j in prange(length):
            x_conv[i] += weights[j] * x_pad[i + (j * dilation)]
    x_conv += bias
    return x_conv  

@njit(parallel=True)
def apply_one_kernel_all_sample(X, weights, length, bias,
                                dilation, padding):
    """
    Apply the convolution operation given the parameters of a convolutional kernel
    and input time series.

    Parameters
    ----------
    x : array, shape = (n_samples, n_features, n_timestamps)
        Time series to transform.
    weights : array, shape = (length)
        Weights of the kernel.
    length : int
        Length of the kernel
    bias : float
        Bias of the kernel
    dilation : int
        Dilation of the kernel
    padding : int
        Padding of the kernel

    Returns
    -------
    X_conv : (n_samples, n_features, n_convolution)
        The convolved input time series.

    """
    n_samples, n_features, n_timestamps = X.shape
    n_conv = n_timestamps - ((length - 1) * dilation) + (2 * padding)
    X_conv = np.zeros((n_samples, n_features, n_conv),dtype=np.float32)
    for i in prange(n_samples):
        for j in prange(n_features):
            X_conv[i,j,:] = apply_one_kernel_one_sample(X[i, j], n_timestamps,
                                                        weights, length, bias, 
                                                        dilation, padding)
    return X_conv


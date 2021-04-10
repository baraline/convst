# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:11:14 2021

@author: Antoine
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from numba import njit, prange
from sktime.utils.data_processing import from_nested_to_3d_numpy, is_nested_dataframe


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

class kernel(BaseEstimator, TransformerMixin):
    def __init__(self, length=None, bias=None, dilation=None, padding=None, 
                 weights=None, verbose=0, id_ft=0):
        self.length = length
        self.bias = bias
        self.dilation = dilation
        self.padding = padding
        self.weights = weights
        self.id_ft = id_ft
        self.verbose = verbose
    
    def _log(self, msg):
        if self.verbose >= 1:
            print(msg)
            
    def _log2(self, msg):
        if self.verbose >= 2:
            print(msg)
    
    def print_info(self):
        """
        Print the kernel parameters

        Returns
        -------
        None.

        """
        print('\n----------------- KERNEL INFO -----------------')
        print('-- LENGHT : {}'.format(self.length))
        print('-- BIAS : {}'.format(self.bias))
        print('-- DILATION : {}'.format(self.dilation))
        print('-- PADDING : {}'.format(self.padding))
        print('-- WEIGHTS : {}'.format(self.weights))
    
    
    def fit(self, X, y=None, normalise=True):
        self._check_is_init()
        return self
    
    def transform(self, X, normalise=True):
        """
        Apply the kernel through a convolution operation on each time series of the input

        Parameters
        ----------
        X : 3-D array
            A 3 dimensional array of shape (n_samples, n_features, n_timestamps).
        normalise : boolean, optional
            If True, X will be normalised before applying the convolution.
            The default is True.

        Returns
        -------
        3-D array
            The convolved input time series. The array will be of shape 
            (n_samples, n_features, n_conv_timestamps)
        """
        
        self._check_is_init()
        X = self._check_array(X)
        if normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        return apply_one_kernel_all_sample(X, self.id_ft, self.weights, 
                                           self.length, self.bias,
                                           self.dilation, self.padding)
                                           

    def _convolve_one_sample(self, x):
        """
        Apply the kernel through a convolution operation to the input time series

        Parameters
        ----------
        x : 1-D array
            A univariate time series.

        Returns
        -------
        1-D array
            The result of the convolution.

        """
        return apply_one_kernel_one_sample(x, x.shape[0], self.weights, 
                                           self.length, self.bias, 
                                           self.dilation, self.padding)  
    
    def _get_indexes(self, convolution_index):
        """
        Return the indexes of the input values used by the convolution to
        generate the value at the convolution_index parameter

        Parameters
        ----------
        convolution_index : int
            An index of the convolution between input and kernel

        Returns
        -------
        1-D array
            An array which contain the indexes of the input values
            used by the convolution operation at convolution_index.

        """
        return np.asarray([convolution_index + (l*self.dilation) - self.padding 
                           for l in range(self.length)])  
    
    def _check_is_init(self):
        if any(self.__dict__[attribute] is None for attribute in ['_length','_bias',
                                                                  '_dilation','_padding',
                                                                  '_weights']):
            raise AttributeError("Kernel attribute not initialised correctly, "
                                 "at least one attribute was set to None")
    
    def _check_array(self, X, coerce_to_numpy=True):
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
    
    @property            
    def length(self): 
        return self._length
    
    @length.setter   
    def length(self, value):
        self._length = value
      
    @property            
    def id_ft(self): 
        return self._id_ft
    
    @id_ft.setter   
    def id_ft(self, value):
        self._id_ft = value
      
    @property            
    def bias(self): 
        return self._bias
    
    @bias.setter   
    def bias(self, value):
        self._bias = value
        
    @property            
    def verbose(self): 
        return self._verbose
    
    @verbose.setter   
    def verbose(self, value):
        self._verbose = value
    
    @property            
    def dilation(self): 
        return self._dilation
    
    @dilation.setter   
    def dilation(self, value):
        self._dilation = value
   
    @property            
    def padding(self): 
        return self._padding
    
    @padding.setter   
    def padding(self, value):
        self._padding = value
    
    @property            
    def weights(self): 
        return self._weights
    
    @weights.setter   
    def weights(self, value):
        self._weights = value
        
class Rocket_feature_kernel(kernel):
    def __init__(self, length=None, bias=None, dilation=None, padding=None,
                 weights=None, id_ft=None, verbose=0, feature_id=None):
        self.kernel_id = feature_id//2
        self.feature_id = feature_id
        super().__init__(length=length, bias=bias, dilation=dilation,
                                    padding=padding, weights=weights,
                                    verbose=verbose, id_ft=id_ft)
        f_val , f_loc = self._get_ft_func()
        self.f_val = f_val
        self.f_loc = f_loc
    
    def fit(self):
        self._check_is_init()
        return self
    
    def get_features(self, X, normalise=True):
        self._check_is_init()
        X = self._check_array(X)
        return self.f_val(self.transform(X, normalise=normalise)[:,self.id_ft])
    
    def get_locs(self, X, normalise=True):
        self._check_is_init()
        X = self._check_array(X)
        X_conv = self.transform(X, normalise=normalise)
        return np.asarray([self.f_loc(x) for x in X_conv[:,self.id_ft]],dtype='object')
    
    def _get_ft_func(self):
        if self.feature_id % 2 == 0:
            if self.bias >=0:
                return self._ft_ppv, self._ft_pnv_loc
            else:
                return self._ft_ppv, self._ft_ppv_loc
        else:
            return self._ft_max, self._ft_max_loc
        
    def _ft_max(self, X_conv):
        return np.max(X_conv, axis=-1)
    
    def _ft_ppv(self, X_conv):
        return np.mean(X_conv > 0, axis=-1)
    
    def _ft_max_loc(self, conv):
        return (conv == conv.max()).nonzero()[0]
    
    def _ft_ppv_loc(self, conv):
        return (conv > 0).nonzero()[0]

    def _ft_pnv_loc(self, conv):
        return (conv <= 0).nonzero()[0]

    
    @property            
    def f_val(self): 
        return self._f_val
    
    @f_val.setter   
    def f_val(self, value):
        self._f_val = value
        
    @property            
    def f_loc(self): 
        return self._f_loc
    
    @f_loc.setter   
    def f_loc(self, value):
        self._f_loc = value
    
    @property            
    def feature_id(self): 
        return self._feature_id
    
    @feature_id.setter   
    def feature_id(self, value):
        self._feature_id = value
    
    @property            
    def kernel_id(self): 
        return self._kernel_id
    
    @kernel_id.setter   
    def kernel_id(self, value):
        self._kernel_id = value
        
class Rocket_kernel(kernel):
    def __init__(self, length=None, bias=None, dilation=None, padding=None,
                 weights=None, verbose=0, id_ft=None, kernel_id=None):
        self.kernel_id = kernel_id
        super().__init__(length=length, bias=bias, dilation=dilation,
                                    padding=padding, weights=weights,
                                    verbose=verbose, id_ft=id_ft)
        
    def fit(self, X, y=None, normalise=True):
        self._check_is_init()
        return self
    
    def get_features(self, X, normalise=True):
        X_conv = self.transform(X, normalise=normalise)
        return np.concatenate((np.mean(X_conv > 0, axis=-1),
                               np.max(X_conv, axis=-1)),axis=1)
    @property            
    def kernel_id(self): 
        return self._kernel_id
    
    @kernel_id.setter   
    def kernel_id(self, value):
        self._kernel_id = value
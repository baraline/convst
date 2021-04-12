# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:10:58 2021

@author: Antoine
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sktime.utils.data_processing import from_nested_to_3d_numpy, is_nested_dataframe
from matplotlib import pyplot as plt

from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, min_dist_shp_loc, generate_strides_1D


class Convolutional_shapelet(BaseEstimator, TransformerMixin):
    """
    A Convolutional Shapelet transformer. It takes an array of values, with 
    dilation and padding parameters to slide itself on input time series.
    The values returned by the transformation is the minimum z-normalised euclidean
    distance from the shapelet to each sample.
    
    Attributes
    ----------
    values : array
        Values of the shapelet, the values should be z-normalised.
        The default is None.
    dilation : int
        Dilation parameter applied when computing distance to input subsequences. 
        The default is None.
    padding : int
        Padding parameter applied when computing distance to input subsequences. 
        The default is None.
    input_ft_id : int
        Identifier of the time series feature on which to apply this shapelet.
        The default is None.
    ft_kernel_id : int
        Identifier of the feature kernel that generated this shapelet.
        The default is None.
    """    
    def __init__(self, values=None, dilation=None, padding=None,
                 input_ft_id=None, ft_kernel_id=None):
        self.values = values
        self.dilation = dilation
        self.padding = padding
        self.input_ft_id = input_ft_id
        self.ft_kernel_id = ft_kernel_id
        
    def fit(self, X, y=None):
        self._check_is_init()
        X = self._check_array(X)
        return self
    
    def transform(self, X, padding_matching=True):
        """
        Transform the input into distance to the Shapelet to be used as a 
        single feature. The distance used is the normalised euclidean distance
        and the minimum distance between the shapelet values and the sample is
        returned for each sample in X.

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        padding_matching : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        array, shape=(X.shape[0])
            Minimum z-normalized euclidean distance from sample to input.

        """
        self._check_is_init()
        X = self._check_array(X)
        n_samples, _, n_timestamps = X.shape
        padding = self.padding
        if not padding_matching:
            padding = 0
            
        if padding > 0:
            X_pad = np.zeros((n_samples, n_timestamps+2*padding))
            X_pad[:,padding:-padding] = X[:,0,:]
        else:
            X_pad = X[:,0,:]   
        X_strides = generate_strides_2D(X_pad,self.values.shape[0],self.dilation)
        X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
                    X_strides.std(axis=-1, keepdims=True) + 1e-8)
        return compute_distances(X_strides, self.values.reshape(1,-1))
        
        
    def _locate(self, x, return_dist=False, return_scale=False, 
                padding_matching=False):        
        min_dist, i_loc = min_dist_shp_loc(generate_strides_1D(x,self.values.shape[0],self.dilation), self.values)
        
        # If padding is used, to get matching in original input (not padded) apply -padding
        loc = np.asarray([i_loc + (j*self.dilation) for j in range(self.values.shape[0])])
        if return_dist and return_scale:
            return loc, min_dist, np.mean(x[loc]), np.std(x[loc])
        elif return_scale:
             return loc, np.mean(x[loc]), np.std(x[loc])
        elif return_dist:
            return loc, min_dist
        else:
            return loc    
    
    
    def _check_is_init(self):
        if any(self.__dict__[attribute] is None for attribute in ['_values',
                                                                  '_dilation',
                                                                  '_padding']):
            raise AttributeError("Shapelet attribute not initialised correctly, "
                                 "at least one attribute was set to None")
    
    def _check_array(self, X, coerce_to_numpy=True):
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
            # convert pd.DataFrame
            if coerce_to_numpy:
                X = from_nested_to_3d_numpy(X)
        return X
    
    
    def plot_loc(self, x, padding_matching=False, ax=None, 
                 alpha=0.75, size=15, color='black', c_x='blue'):
        """
        Plot the shapelet on the input. The shapelet will be displayed on its
        closest match to the input serie and scaled to the input. 
        If padding_matching is used, the padded version of x will be displayed.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        padding_matching : TYPE, optional
            DESCRIPTION. The default is False.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        alpha : TYPE, optional
            DESCRIPTION. The default is 0.75.
        size : TYPE, optional
            DESCRIPTION. The default is 15.
        color : TYPE, optional
            DESCRIPTION. The default is 'black'.
        c_x : TYPE, optional
            DESCRIPTION. The default is 'blue'.

        Returns
        -------
        None.

        """
        padding = self.padding
        if not padding_matching:
            padding = 0
            
        if padding > 0:
            x_pad = np.zeros(x.shape[0] + 2 * padding)
            x_pad[padding:-padding] = x
        else:
            x_pad = x
        
        loc, mean, std = self._locate(x_pad, return_dist=False, return_scale=True,
                                     padding_matching=padding_matching)
        vals = (self.values * std) + mean
        padding = self.padding
        if ax is None:
            plt.plot(x_pad,c=c_x)
            plt.scatter(loc,vals,alpha=alpha,color=color,s=size)            
            plt.show()
        else:
            ax.plot(x_pad,c=c_x)
            ax.scatter(loc,vals,alpha=alpha,color=color,s=size)
    
    
    @property            
    def values(self):
        return self._values
      
    @values.setter   
    def values(self, value):        
        self._values = value
              
    @property            
    def padding(self):
        return self._padding
      
    @padding.setter   
    def padding(self, value):
        if type(value) is not int:
            value = int(value)
        self._padding = value
    
    @property            
    def dilation(self):
        return self._dilation
      
    @dilation.setter   
    def dilation(self, value):
        if type(value) is not int:
            value = int(value)
        self._dilation = value    
    
    @property            
    def input_ft_id(self):
        return self._input_ft_id
      
    @input_ft_id.setter   
    def input_ft_id(self, value):        
        self._input_ft_id = value
        
    @property            
    def ft_kernel_id(self):
        return self._ft_kernel_id
      
    @ft_kernel_id.setter   
    def ft_kernel_id(self, value):        
        self._ft_kernel_id = value
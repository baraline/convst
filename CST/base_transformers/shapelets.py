# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:10:58 2021

@author: Antoine
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from cst.utils.checks_utils import check_array_3D, check_array_1D
from cst.utils.shapelets_utils import generate_strides_2D


class Convolutional_shapelet(BaseEstimator, TransformerMixin):

    def __init__(self, values=None, dilation=None):
        """
        A Convolutional Shapelet transformer. It takes an array of values, with a
        dilation parameter to slide itself on input time series.
        The values returned by the transformation is the minimum z-normalised 
        squarred euclidean distance from the shapelet to each sample.
    
        Attributes
        ----------
        values : array, shape = (length,)
            Values of the shapelet, those values will be z-normalised.
            
        dilation : int
            Dilation parameter applied when computing distance to input subsequences. 
            
        """
        self.values = values
        self.dilation = dilation

    def fit(self, X, y=None):
        """
        Placeholder method to allow fit_transform method.

        Parameters
        ----------
        X : ignored
        
        y : ignored, optional


        Returns
        -------
        self

        """
        self._check_is_init()        
        return self

    def transform(self, X, return_location=False, return_scale=False):
        """
        Transform the input using Shapelet distance (minimum z-normalized
        squarred euclidean distance to all subsequences of a time series)

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input time series.
            
        return_location : boolean, optional
            Also return the location of the minimum distance. 
            Default is False.
            
        return_scale : boolean, optional
            Also return the scale (mean and std) of the subsequence of inputs were
            minimum distance was found. Default is False.

        Returns
        -------
        array, shape=(n_samples, n_features)
            Shapelet distance to all inputs and features.

        """
        self._check_is_init()
        X = check_array_3D(X)
        n_samples, n_features, n_timestamps = X.shape
        
        if return_location:
            locs = np.zeros((n_samples, n_features))
        if return_scale:
            scales = np.zeros((n_samples, n_features, 2))
        dist = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            X_strides = generate_strides_2D(
                X[:,i], self.length, self.dilation)
            X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
                X_strides.std(axis=-1, keepdims=True) + 1e-8)
            print(X_strides.shape)
            for j in range(X.shape[0]):
                d = cdist(X_strides[j], self.values.reshape(1,-1), metric='sqeuclidean')
                dist[j, i] += d.min(axis=0)
                if return_location:
                    locs[j, i] += d.argmin(axis=0)
                if return_scale:
                    subseq = X[j, i,
                               np.asarray([d.argmin(axis=0) + j*self.dilation 
                                           for j in range(self.length)])
                               ]
                    scales[j, i, 0] += subseq.mean()
                    scales[j, i, 1] += subseq.std()
                
        if return_location and return_scale:
            return dist, locs, scales
        elif return_location:
            return dist, locs
        elif return_scale:
            return dist, scales
        else:
            return dist

    def _check_is_init(self):
        if any(self.__dict__[attribute] is None for attribute in ['_values',
                                                                  '_dilation']):
            raise AttributeError("Shapelet attribute not initialised correctly, "
                                 "at least one attribute was set to None")

    def plot(self, X, ax=None,
                 alpha=0.75, x_alpha=0.9, lw=3, x_lw=4,
                 color='red', c_x='blue'):
        """
        Plot the shapelet on an input time series. The shapelet will be displayed on its
        closest match to the input serie and scaled to the input. 

        Parameters
        ----------
        X : array, shape=(n_timestamps)
            Input time series.
        ax : matplotlib.axes, optional
            A matplotlib axe on which to plot.
        alpha : float, optional
            The alpha parameter for the shapelet. The default is 0.75.
        x_alpha : float, optional
            The alpha parameter for the input. The default is 0.9.
        x_lw : float, optional
            The linewidth of the input. The default is 4.
        lw : float, optional
            The linewidth of the Shapelet. The default is 3.
        color : TYPE, optional
            Color of the shapelet. The default is 'black'.
        c_x : string, optional
            Color of the input time serie. The default is 'blue'.

        Returns
        -------
        None.

        """
        _, loc, scale = self.transform(X.reshape(1,1,-1), return_scale=True,
                                        return_location=True)
        vals = (self.values * scale[0,0,1]) + scale[0,0,0]
        loc = [loc[0,0] + j*self.dilation for j in range(self.length)]
        print(vals)
        print(loc)
        if ax is None:
            plt.plot(X, c=c_x, alpha=x_alpha, linewidth=x_lw)
            plt.plot(loc, vals, alpha=alpha, color=color, linestyle='dashed', linewidth=lw)
            plt.show()
        else:
            ax.plot(X, c=c_x, alpha=x_alpha, linewidth=x_lw)
            ax.plot(loc, vals, alpha=alpha, color=color, linestyle='dashed', linewidth=lw)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        value = check_array_1D(value)
        self._values = value
        self.length = self.values.shape[0]
        
    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    @property
    def dilation(self):
        return self._dilation

    @dilation.setter
    def dilation(self, value):
        if type(value) is not int:
            value = int(value)
        self._dilation = value


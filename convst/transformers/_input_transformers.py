#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:35:58 2022

@author: lifo
"""
import numpy as np


from convst.utils.checks_utils import check_array_3D

from numba import njit, prange

from pyts.approximation import DiscreteFourierTransform, SymbolicAggregateApproximation

from scipy.signal import periodogram
from scipy.fft import fht, fhtoffset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class c_StandardScaler(StandardScaler):
    def fit(self, X, y=None):
        self.usefull_atts = np.where(np.std(X, axis=0) != 0)[0]
        return super().fit(X[:, self.usefull_atts], y=y)
    
    def transform(self, X):
        return super().transform(X[:, self.usefull_atts])
    
class c_MinMaxScaler(MinMaxScaler):
    def fit(self, X, y=None):
        self.usefull_atts = np.where(np.std(X, axis=0) != 0)[0]
        return super().fit(X[:, self.usefull_atts], y=y)
    
    def transform(self, X):
        return super().transform(X[:, self.usefull_atts])

@njit(cache=True)
def z_norm_one_sample(x):
    n_features, n_timestamps = x.shape
    x_new = np.empty((n_features, n_timestamps))
    for i in prange(n_features):
        x_new[i] = (x[i] - x[i].mean()) / (x[i].std() + 1e-8)
    return x_new

@njit(cache=True, parallel=True)
def z_norm_all_samples(X):
    n_samples, n_features, n_timestamps = X.shape
    X_new = np.empty((n_samples, n_features, n_timestamps))
    for i in prange(n_samples):
        X_new[i] = z_norm_one_sample(X[i])
    return X_new

class Z_normalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_univariate=False
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = check_array_3D(X, is_univariate=self.is_univariate)
        return z_norm_all_samples(X)
    
class Raw(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X

class Derivate(BaseEstimator, TransformerMixin):
    def __init__(self, order=1, random=False):
        self.order = order
        self.random = random
        
    def fit(self, X, y=None):
        if self.random:
            self._random_init()
        return self
    
    def transform(self, X):
        for i in range(self.order):
            X = np.diff(X)
        return X

    def _random_init(self):
        self.set_params(**{"order":np.random.choice(np.arange(1,5))})

class Periodigram(BaseEstimator, TransformerMixin):
    def __init__(self, window_type="boxcar", random=False):
        self.window_type = window_type
        self.random = random
        self.is_univariate=False
        
    def fit(self, X, y=None):
        if self.random:
            self._random_init()
        return self
    
    def transform(self, X):
        n_ts = periodogram(X[0,0,:], detrend=False, window=self.window_type)[1].shape[0]
        X_new = np.empty((X.shape[0], X.shape[1], n_ts))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_new[i,j] = periodogram(X[i,j], detrend=False, window=self.window_type)[1]
        return X_new
    
    def _random_init(self):
        self.set_params(**{"window_type":np.random.choice(self._get_windows())})
    
    def _get_windows(self):
        return np.asarray(
            ["boxcar","triang","blackman","hamming","hann",
             "bartlett","flattop","parzen","bohman",
             "blackmanharris","nuttall","barthann",
             "cosine","exponential","tukey","taylor"]
        )

#TODO : adapt to multivariate/uneven length context
class Sax(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=10, strategy="uniform", random=False):
        self.random = random
        self.n_bins = n_bins
        self.strategy = strategy
        
    
    def fit(self, X, y=None):
        if self.random:
            self._random_init(X.shape[1])
        self.transformer = SymbolicAggregateApproximation(
            n_bins=self.n_bins, strategy=self.strategy, alphabet='ordinal'
        )
        self.transformer.fit(X[:,0,:])
        return self
    
    def transform(self, X):
        X = self.transformer.transform(X[:,0,:])
        return X[:, np.newaxis, :]
    
    def _random_init(self, n_timestamps):
        self.set_params(**{"n_bins":np.random.choice(np.arange(2,min(n_timestamps,26)))})

#TODO : adapt to multivariate/uneven length context
class FourrierCoefs(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_coefs=None, drop_sum=False, anova=False, norm_mean=False,
        norm_std=False
    ):
        self.n_coefs = n_coefs
        self.drop_sum = drop_sum
        self.anova = anova
        self.norm_mean = norm_mean
        self.norm_std = norm_std
    
    def fit(self, X, y=None):
        self.transformer = DiscreteFourierTransform(
            n_coefs=self.n_coefs, drop_sum=self.drop_sum, anova=self.anova,
            norm_std=self.norm_std, norm_mean=self.norm_mean,
        )
        self.transformer.fit(X[:,0,:], y=y)
        return self
    
    def transform(self, X):
        X = self.transformer.transform(X[:,0,:])
        return X[:, np.newaxis, :]


class FastHankelTransform(BaseEstimator, TransformerMixin):
    def __init__(
        self, dln=0.01, mu=1, offset=0.0, bias=0.0, use_optimal_offset=True
    ):
        self.dln = dln
        self.mu = mu
        self.offset = offset
        self.bias = bias
        self.use_optimal_offset = use_optimal_offset
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = np.zeros(X.shape)
        for i in range(X.shape[0]):
            if self.use_optimal_offset:
                X_new[i] = fht(
                    X[i], self.dln, self.mu,
                    offset=fhtoffset(self.dln, self.dln, self.bias),
                    bias=self.bias
                )
            else:
                X_new[i] = fht(
                    X[i], self.dln, self.mu, offset=self.offset,
                    bias=self.bias
                )
        return X_new
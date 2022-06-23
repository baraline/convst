# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:32:33 2022

@author: a694772
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from numba import njit, prange
import numpy as np

@njit(cache=True, fastmath=True)
def de(x,y):
    _s = 0
    #x = (x - x.mean())/(x.std()+1e-8)
    #y = (y - y.mean())/(y.std()+1e-8)
    for i in prange(x.shape[0]-1):
        _s += abs(x[i] - y[i])
    return _s

@njit(cache=True, fastmath=True)
def d(x,y):
    _s = 0
    #x = (x - x.mean())/(x.std()+1e-8)
    #y = (y - y.mean())/(y.std()+1e-8)
    for i in prange(x.shape[0]-1):
        _s += abs(x[i] - y[i]) + abs(x[i+1]-y[i]) + abs(y[i+1]-x[i])
    return _s
    
@njit(cache=True, parallel=True)
def _transform(X, Y):
    d_mat = np.zeros((X.shape[0],Y.shape[0]))
    for i in prange(X.shape[0]):
        for j in prange(Y.shape[0]):
            d_mat[i,j] = d(X[i],Y[j])
    return d_mat


class Knn(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_neighbors, metric='precomputed'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        
    def fit(self, X, y):
        self.inputs = X[:,0]
        self.c_inputs = y
        return self
    
    def predict(self, X):
        X = _transform(self.inputs,X[:,0])
        print(X.shape)
        print(X.argmin(axis=0).shape)
        p = self.c_inputs[X.argmin(axis=0)]
        return p


k = Knn(1).fit(X_train,y_train)
print(k.score(X_test, y_test))
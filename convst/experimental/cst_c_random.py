# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:00:13 2021
DSTEC
@author: a694772
"""
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from convst.transformers import Raw, R_DST, FourrierCoefs, Derivate
from joblib import Parallel, delayed
from numba import set_num_threads

def _parallel_transformer_fit(transformer, X, y, n_thread_per_job):
    set_num_threads(n_thread_per_job)
    return transformer.fit(X, y)

def _parallel_transformer_transform(transformer, X, n_thread_per_job):
    set_num_threads(n_thread_per_job)
    return transformer.transform(X)

class CST_C_Random(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 input_dict = {
                 '0':
                    {'transformer':Raw(),'n_shapelets':8000,
                     'p_norm':0.9, 'shapelet_sizes':[7,9,11]},
                 '1':
                   {'transformer':FourrierCoefs(),'n_shapelets':1000,
                    'p_norm':0.1, 'shapelet_sizes':[3,5,7]}, 
                 '2':
                   {'transformer':Derivate(),'n_shapelets':1000,
                    'p_norm':0.75, 'shapelet_sizes':[7,9,11]}, 
                 },
                 random_state=None,
                 n_jobs=1, n_thread_per_job=1,
                 BaseClassifier=None,
                 ):
        
        self.input_dict = input_dict
        self.random_state = random_state
        if BaseClassifier is None :
            self.BaseClassifier = make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(alphas=np.logspace(-6,6,20))
            )
        else:
            self.BaseClassifier = BaseClassifier
        self.n_jobs = n_jobs
        self.n_thread_per_job = n_thread_per_job
    
    def _initialize_transformer_pool(self):
        keys = list(self.input_dict.keys())
        n_trans = len(keys)
        transformers = np.empty(n_trans, dtype=object)
        for i in range(n_trans):
            key = keys[i]
            transformers[i] = make_pipeline(
                clone(self.input_dict[key]['transformer']),
                clone(R_DST(n_shapelets=self.input_dict[key]['n_shapelets'],
                          shapelet_sizes=self.input_dict[key]['shapelet_sizes'],
                          p_norm=self.input_dict[key]['p_norm']))
            )
        return transformers
    
    def fit(self, X, y):
        transformers = self._initialize_transformer_pool()
        transformers = Parallel(
            n_jobs=self.n_jobs,
        )(
            delayed(_parallel_transformer_fit)(
                e, X, y, self.n_thread_per_job
            )
            for e in transformers
        )          
        self.transformers = transformers
        X = self.transform(X)
        self.BaseClassifier.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, ['BaseClassifier', 'transformers'])
        X_new = Parallel(
            n_jobs=self.n_jobs,
        )(
            delayed(_parallel_transformer_transform)(
                e, X, self.n_thread_per_job
            )
            for e in self.transformers
        )
        return np.concatenate(X_new, axis=1)
        
    def predict(self, X, alpha=4):
        check_is_fitted(self, ['BaseClassifier', 'transformers'])
        X = self.transform(X)
        return self.BaseClassifier.predict(X)


# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:00:13 2021

@author: a694772

from sklearn.metrics import accuracy_score, make_scorer

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier

from convst.transformers import Raw, CST_Random
from joblib import Parallel, delayed
from numba import set_num_threads

#from convst.classifiers.rotationforest import RotationForest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

def _parallel_fit(estimator, X, y, n_thread_per_job, nn_cv=None):
    set_num_threads(n_thread_per_job)
    X_cv = estimator[:-1].fit_transform(X, y)
    estimator[-1].fit(X_cv, y)
    n_cv = max(2,np.bincount(y).max())
    if nn_cv:
        score = np.mean(cross_val_score(
            clone(estimator[-1]), X_cv, y, cv=min(n_cv,nn_cv),
            scoring=make_scorer(accuracy_score), 
            n_jobs=min(n_cv, n_thread_per_job))
        )
    else:
        score = 1       
    return estimator, score
    
def _parallel_predict(estimator, X, n_thread_per_job):
    set_num_threads(n_thread_per_job)
    return estimator.predict_proba(X)

class RidgeClassifierCV_Proba(RidgeClassifierCV):
    def predict_proba(self, X):
        d = self.decision_function(X)
        n_classes = self.classes_.size
        d = np.exp(d) / (1+np.exp(d))
        if n_classes == 2:
            return np.vstack((1-d, d)).T
        return d

class CST_EC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_shapelets=10000, transformations=[Raw()], 
                 shapelet_sizes=[7,9,11], random_state=None,
                 n_jobs=1, n_thread_per_job=1,
                 P=98, add_padding=True, BaseClassifier=None,
                 z_norm_input=True):
        self.n_shapelets = n_shapelets
        self.shapelet_sizes = shapelet_sizes
        self.random_state = random_state
        if BaseClassifier is None :
            self.BaseClassifier = RandomForestClassifier(
                n_estimators=200, ccp_alpha=0.005, n_jobs=n_thread_per_job
            )
            
        else:
            self.BaseClassifier = BaseClassifier
        self.transformations = np.asarray(transformations)
        self.n_jobs = n_jobs
        self.n_thread_per_job = n_thread_per_job
        self.P = P
        self.add_padding = add_padding
        self.z_norm_input = z_norm_input
    
    def _initialize_estimator_pool(self):
        n_trans = self.transformations.size
        estimators = np.empty(n_trans, dtype=object)
        n_shapelets_per_estimator = self.n_shapelets // n_trans
        for i in range(n_trans):
            estimators[i] = make_pipeline(
                clone(self.transformations[i]),
                clone(CST_Random(n_shapelets=n_shapelets_per_estimator,
                          shapelet_sizes=self.shapelet_sizes,
                          P = self.P, z_norm_input=self.z_norm_input,
                          add_padding = self.add_padding)),
                clone(self.BaseClassifier)
            )
        return estimators
    
    def fit(self, X, y, n_cv=10):
        estimators = self._initialize_estimator_pool()
        estimators = np.asarray(Parallel(
            n_jobs=self.n_jobs,
        )(
            delayed(_parallel_fit)(
                e, X, y, self.n_thread_per_job, nn_cv=n_cv
            )
            for i, e in enumerate(estimators)
        ),dtype=object)
            
        self.estimators_ = estimators[:,0]
        self.cv_scores_ = estimators[:,1].astype(float)
        return self

    def predict(self, X, alpha=4):
        check_is_fitted(self, ['estimators_', 'cv_scores_'])
        preds = np.asarray(Parallel(
            n_jobs=self.n_jobs,
        )(
            delayed(_parallel_predict)(
                e, X, self.n_thread_per_job
            )
            for i, e in enumerate(self.estimators_)
        ))
        preds = preds ** alpha
        #n_estim
        for i in range(preds.shape[0]):
            preds[i,:] *= self.cv_scores_[i]
        return np.argmax(preds.sum(axis=0),axis=1)

    def get_shapelet_importances(self):
        pass        
"""
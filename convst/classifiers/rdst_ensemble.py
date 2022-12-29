# -*- coding: utf-8 -*-

import numpy as np

from joblib import Parallel

from convst.utils.checks_utils import (check_n_jobs, check_array_1D,
                                       check_is_boolean, check_is_numeric)
from convst.transformers import R_DST
from convst.transformers._input_transformers import (
    Raw, Derivate, Periodigram
)
from sklearn.utils.fixes import delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils.extmath import softmax
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from numba import set_num_threads

class _internalRidgeCV(RidgeClassifierCV): 
    def __init__(self, **kwargs):
        super().__init__(
            store_cv_values=True,
            scoring=make_scorer(accuracy_score),
            alphas=np.logspace(-4,4,20),
            **kwargs
        )
    
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        return super().fit(self.scaler.transform(X), y)
    
    def predict(self, X):
        return super().predict(self.scaler.transform(X))
    
    def predict_proba(self, X):
        d = self.decision_function(self.scaler.transform(X))
        if len(d.shape) == 1:
            d = np.c_[-d, d]
        return softmax(d)
    
    def get_loocv_train_acc(self, y_train):
        if self.store_cv_values:
            alpha_idx = np.where(self.alphas == self.alpha_)[0]
            cv_vals = self.cv_values_[:,:,alpha_idx]
            if cv_vals.shape[1] == 1:
                return accuracy_score((cv_vals[:,0]>0).astype(int), y_train)
            else:
                return accuracy_score(cv_vals.argmax(axis=1), y_train)
        else:
            raise ValueError('LOOCV training accuracy is only available with store_cv_values to True')


def _parallel_fit(X, y, model):
    return model.fit(X, y)

def _parallel_predict(X, model, w):
    return model.predict_proba(X) * w

class R_DST_Ensemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_shapelets_per_estimator=10000,
        shapelet_lengths=[11],
        shapelet_lengths_bounds=None,
        lengths_bounds_reduction=0.5,
        prime_dilations=False,
        n_samples=None,
        n_jobs=1,
        prefer=None,
        require='sharedmem',
        random_state=None,
        shp_alpha=0.5,
        a_w=4,
        proba_norm=[0.8, 0.8, 0.8],
        phase_invariance=False,
        input_transformers=None
    ):
        self.n_shapelets_per_estimator = check_is_numeric(n_shapelets_per_estimator)
        self.shapelet_lengths = check_array_1D(shapelet_lengths)
        self.n_jobs = n_jobs
        if shapelet_lengths_bounds is not None:
            self.shapelet_lengths_bounds = check_array_1D(shapelet_lengths_bounds)
        else:
            self.shapelet_lengths_bounds = shapelet_lengths_bounds
        self.lengths_bounds_reduction = check_is_numeric(lengths_bounds_reduction)
        self.prime_dilations = check_is_boolean(prime_dilations)
        self.prefer = prefer
        self.require = require
        self.random_state = random_state
        if shapelet_lengths_bounds is not None:
            self.n_samples = check_is_numeric(n_samples)
        else:
            self.n_samples = n_samples
        self.shp_alpha = check_is_numeric(shp_alpha)
        self.a_w = check_is_numeric(a_w)
        self.proba_norm = check_array_1D(proba_norm)
        self.phase_invariance = check_is_boolean(phase_invariance)
        
        if input_transformers is None:
            self.input_transformers = [
                Raw(),
                Derivate(),
                Periodigram()
            ]
        else:
            self.input_transformers = input_transformers
        
        if len(self.input_transformers) != len(self.proba_norm):
            raise ValueError(
                'The length of proba norm array should be equal to the'
                ' length of the input transformers array, but found '
                '{} for proba_norm and {} for input transformers'.format(
                 len(self.proba_norm), len(self.input_transformers))
            )
        
    def _more_tags(self):
        return {
            "capability:variable_length": True,
            "capability:univariate": True,
            "capability:multivariate": True
        }
        
    def _manage_n_jobs(self):
        total_jobs = check_n_jobs(self.n_jobs)
        self.n_jobs = min(len(self.input_transformers),total_jobs)
        self.n_jobs_rdst = max(1,total_jobs//self.n_jobs)
        
    def fit(self, X, y):
        self._manage_n_jobs()
        set_num_threads(self.n_jobs_rdst)
        models = Parallel(
            n_jobs=self.n_jobs,
            prefer=self.prefer,
            require=self.require
        )(
            delayed(_parallel_fit)(
                X, y, 
                make_pipeline(
                    self.input_transformers[i],
                    R_DST(
                        n_shapelets=self.n_shapelets_per_estimator,
                        alpha=self.shp_alpha, n_samples=self.n_samples, 
                        proba_norm=self.proba_norm[i], n_jobs=False,
                        shapelet_lengths=self.shapelet_lengths,
                        phase_invariance=self.phase_invariance,
                        prime_dilations=self.prime_dilations,
                        shapelet_lengths_bounds=self.shapelet_lengths_bounds,
                        lengths_bounds_reduction=self.lengths_bounds_reduction
                    ),
                    _internalRidgeCV()
                )
            )
            for i in range(len(self.input_transformers))
        )
            
        self.models = models
        self.model_weights = [model['_internalridgecv'].get_loocv_train_acc(y)**self.a_w
                              for model in models]
        
        return self


    def predict(self, X):
        preds_proba = Parallel(
            n_jobs=self.n_jobs,
            prefer=self.prefer,
            require=self.require
        )(
            delayed(_parallel_predict)(
                X,
                self.models[i],
                self.model_weights[i]
            )
            for i in range(len(self.model_weights))
        )
        #n_samples, n_models
        preds_proba = np.asarray(preds_proba).sum(axis=0)
        return preds_proba.argmax(axis=1)

    
        
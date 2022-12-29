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

from sklearn.utils.validation import check_is_fitted, check_random_state

from numba import set_num_threads

class _internalRidgeCV(RidgeClassifierCV): 
    def __init__(self, **kwargs):
        super().__init__(
            store_cv_values=True,
            scoring=make_scorer(accuracy_score),
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
    """
    
    Parameters
    ----------
    transform_type : str, optional
        Type of transformer to use. Based on the characteristics of the input
        time series, different class of transformer must be used, for example
        the tranformer for univariate series is not the same as for
        multivariate ones for run-time optimization reasons.
        The default is 'auto', which automatically select the transformer based
        on the data passed in the fit method.
    phase_invariance : bool, optional
        Wheter to use phase invariance for shapelet sampling and distance 
        computation. The default is False.
    distance : str, optional
        The distance function to use whe computing distances between shapelets
        and time series. Choose between 'euclidean','manhattan' and 'squared_euclidean'.
        The default is 'manhattan'.
    alpha : float, optional
        The alpha similarity parameter, the higher the value, the lower the 
        allowed number of common indexes with previously sampled shapelets 
        when sampling a new one with similar parameters. It can cause the
        number of sampled shapelets to be lower than n_shapelets if the
        whole search space has been covered. The default is 0.5.
    normalize_output : boolean, optional
        Wheter to normalize the argmin and shapelet occurrence feature by the 
        length of the series from which it was extracted. This is mostly useful
        for variable length time series. The default is False.
    n_samples : float, optional
        Proportion (in ]0,1]) of samples to consider for the shapelet
        extraction. The default is None, meaning that all samples are used.
    n_shapelets : int, optional
        The maximum number of shapelet to be sampled. The default is 10_000.
    shapelet_lengths : array, optional
        The set of possible length for shapelets. The values can be integers
        to specify an absolute length, or a float, to specify a length relative 
        to the input time series length. The default is [11].
    shapelet_lengths_bounds : array, optional
        An 1D array with two elements containing the min and max possible 
        length for shapelet candidate, can be int or float. The default is
        None, meaning that shapelet_lengths parameter is used.
    lengths_bounds_reduction : float, optional
        A float in ]0,1], quantifying the proportion of lengths to explore 
        between the min and max bounds of shapelet_lengths_bounds. The default
        is 0.5. For example, with bounds as [4,10], and a reduction of 0.5,
        only [4,6,8,10] will be considered as possible lengths.
    prime_dilations : bool, optional
        If True, only dilation with prime values will be considered for 
        shapelet candidates. This will greatly speed-up the algorithm
        for long time series and/or short shapelet length, possibly at the cost
        of some accuracy.
    proba_norm : float, optional
        The proportion of shapelets that will use a normalized distance 
        function, which induce scale invariance. The default is 0.8.
    percentiles : array, optional
        The two perceniles used to select the lambda threshold used to compute
        the Shapelet Occurrence feature. The default is [5,10].
    n_jobs : int, optional
        The number of threads used to sample and compute the distance vectors.
        The default is 1, -1 means all available cores.
    random_state : object, optional
        The seed for the random state. The default is None.
    max_channels : int, optional
        The maximum number of feature possibly considered by a multivariate 
        shapelet. The default is None, meaning max_chanels=n_features.
    min_len : int, optional
        The minimum length of an input time series for variable length input.
        The default is None, meaning min_len=min(n_timestamps) on the training data.
        This can cause error if a shorter serie sis present in the test set.
    class_weight : object, optional
         Class weight option of Ridge Classifier, either None, "balanced" or a
         custom dictionnary of weight for each class. The default is None.
    fit_intercept : bool, optional
         If True, the intercept term will be fitted during the ridge regression.
         The default is True.
    alphas : array, optional
         Array of alpha values to try which influence regularization strength, 
         must be a positive float. The default is np.logspace(-4,4,20).
    """
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
        input_transformers=None,
        transform_type='auto',
        distance='manhattan',
        normalize_output=False,
        percentiles=[5,10],
        max_channels=None,
        min_len=None,
        class_weight=None,
        fit_intercept=True,
        alphas_ridge=list(np.logspace(-4,4,20))
    ):
        self.transform_type = transform_type
        self.phase_invariance = check_is_boolean(phase_invariance)
        self.distance = distance
        self.normalize_output = check_is_boolean(normalize_output)
        self.n_samples = check_is_numeric(n_samples) if n_samples is not None else n_samples
        self.shapelet_lengths_bounds = shapelet_lengths_bounds
        self.prime_dilations = check_is_boolean(prime_dilations)
        self.percentiles = percentiles
        self.random_state = check_random_state(random_state)
        self.max_channels=max_channels
        self.min_len=min_len
        self.lengths_bounds_reduction = lengths_bounds_reduction
        self.n_shapelets_per_estimator = check_is_numeric(n_shapelets_per_estimator)
        self.shapelet_lengths = check_array_1D(shapelet_lengths)
        self.n_jobs = n_jobs
        
        self.prefer = prefer
        self.require = require
        self.shp_alpha = check_is_numeric(shp_alpha)
        self.a_w = check_is_numeric(a_w)
        self.proba_norm = check_array_1D(proba_norm)
        
        self.alphas_ridge = alphas_ridge
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept
      
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
                        proba_norm=self.proba_norm[i],
                        n_jobs=False,
                        shapelet_lengths=self.shapelet_lengths,
                        phase_invariance=self.phase_invariance,
                        prime_dilations=self.prime_dilations,
                        shapelet_lengths_bounds=self.shapelet_lengths_bounds,
                        lengths_bounds_reduction=self.lengths_bounds_reduction,
                        transform_type=self.transform_type,
                        percentiles=self.percentiles,
                        min_len=self.min_len,
                        max_channels=self.max_channels,
                        random_state=self.random_state
                    ),
                    _internalRidgeCV(
                        alphas=self.alphas_ridge,
                        fit_intercept=self.fit_intercept,
                        class_weight=self.class_weight
                    )
                )
            )
            for i in range(len(self.input_transformers))
        )
            
        self.models = models
        self.model_weights = [model['_internalridgecv'].get_loocv_train_acc(y)**self.a_w
                              for model in models]
        return self


    def predict(self, X):
        check_is_fitted(self, ['model_weights'])
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

    
        
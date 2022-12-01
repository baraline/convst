# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from convst.transformers._input_transformers import c_StandardScaler
from convst.transformers import R_DST

from convst.utils.checks_utils import check_n_jobs
from sklearn.metrics import accuracy_score

from numba import set_num_threads

class R_DST_Ridge(BaseEstimator, ClassifierMixin):
    """
    A wrapper class which use R_DST as a transformer, followed by a Ridge 
    Classifier.
    
    Attributes
    ----------
    classifier : object
        A sklearn pipeline for RidgeClassifierCV with L2 regularization.
    transformer : object
        An instance of R_DST.

    Parameters
    ----------
    n_shapelets : int, optional
        Number of shapelets to generate. The default is 10000.
    shapelet_sizes : array, optional
        An array of int which indicate the possible absolute shapelet sizes.
        Or an array of float which will give shapelet sizes relative to input length.
        The default is [11]
    p_norm : float, optional
        A float between 0 and 1 indicating the proportion of shapelets that
        will use a z-normalized distance. The default is 0.8.
    percentiles : array, shape=(2), optional
        The two percentiles (between 0 and 100) between which the value of the
        threshold will be sampled during shapelet generation. 
        The default is [5,10].
    random_state : int, optional
        Value of the random state for all random number generation.
        The default is None.
    n_jobs : int, optional
        Number of thread used by numba for the computational heavy part
        of the algortihm. The default is 1. Change to -1 to use all
        available cores.
    class_weight : object, optional
        Class weight option of Ridge Classifier, either None, "balanced" or a
        custom dictionnary of weight for each class. The default is None.
    fit_intercept : bool, optional
        If True, the intercept term will be fitted during the ridge regression.
        The default is True.
    alphas : array, optional
        Array of alpha values to try which influence regularization strength, 
        must be a positive float.
        The default is np.logspace(-4,4,10).
    """
    
    def __init__(
        self, 
        transform_type='auto',
        phase_invariance=False,
        distance='manhattan',
        alpha=0.5,
        normalize_output=False,
        n_samples=None,
        n_shapelets=10_000,
        shapelet_lengths=[11],
        shapelet_lengths_bounds=None,
        lengths_bounds_reduction=0.5,
        prime_dilations=False,
        proba_norm=0.8,
        percentiles=[5,10],
        n_jobs=1,
        random_state=None,
        min_len=None,
        class_weight=None, 
        fit_intercept=True,
        alphas_ridge=list(np.logspace(-4,4,20))
    ):
        self.alphas_ridge=alphas_ridge
        self.class_weight=class_weight
        self.fit_intercept=fit_intercept
        self.transform_type=transform_type
        self.phase_invariance=phase_invariance
        self.prime_dilations=prime_dilations
        self.distance=distance
        self.alpha=alpha
        self.normalize_output=normalize_output
        self.n_samples=n_samples
        self.shapelet_lengths_bounds=shapelet_lengths_bounds
        self.lengths_bounds_reduction=lengths_bounds_reduction
        self.n_shapelets=n_shapelets
        self.shapelet_lengths=shapelet_lengths
        self.proba_norm=proba_norm
        self.percentiles=percentiles
        if isinstance(n_jobs, bool):
            self.n_jobs=n_jobs
        else:
            self.n_jobs=check_n_jobs(n_jobs)
            set_num_threads(self.n_jobs)
        self.random_state=random_state
        self.min_len=min_len
    
    def _more_tags(self):
        return {
            "capability:variable_length": True,
            "capability:univariate": True,
            "capability:multivariate": True
        }

    def _init_components(self):
        self.classifier = make_pipeline(
            c_StandardScaler(with_mean=True),
            RidgeClassifierCV(
                alphas=self.alphas_ridge,
                class_weight=self.class_weight, 
                fit_intercept=self.fit_intercept
            )
        )
        self.transformer = R_DST(
            transform_type=self.transform_type,
            phase_invariance=self.phase_invariance,
            distance=self.distance,
            alpha=self.alpha,
            prime_dilations=self.prime_dilations,
            shapelet_lengths_bounds=self.shapelet_lengths_bounds,
            lengths_bounds_reduction=self.lengths_bounds_reduction,
            normalize_output=self.normalize_output,
            n_samples=self.n_samples,
            n_shapelets=self.n_shapelets,
            n_jobs=False,
            shapelet_lengths=self.shapelet_lengths,
            proba_norm=self.proba_norm,
            percentiles=self.percentiles,
            random_state=self.random_state,
            min_len=self.min_len 
        )
    
    def fit(self, X, y):
        """
        Fit method. Random shapelets are generated using the parameters
        supplied during initialisation. Then, input time series are transformed
        using R_DST before classification with a Ridge classifier.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Input time series.
            
        y : array, shape=(n_samples)
            Class of the input time series.
        """
        self._init_components()
        self.transformer = self.transformer.fit(X, y)
        self.classifier = self.classifier.fit(self.transformer.transform(X), y)
        return self
        
    def predict(self, X):
        """
        Transform the input time series with R_DST and predict their classes
        using the fitted Ridge Classifier.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Input time series.

        Returns
        -------
        array, shape=(n_samples)
            Predicted class for each input time series

        """
        check_is_fitted(self, ['classifier'])
        return self.classifier.predict(self.transformer.transform(X))
    
    def score(self, X, y):
        """
        Perform the prediction on input time series and return the accuracy 
        score based on the class information.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Input time series.
            
        y : array, shape=(n_samples)
            Class of the input time series.

        Returns
        -------
        float
            Accuracy score on the input time series

        """
        preds = self.predict(X)
        return accuracy_score(y, preds)

# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from convst.transformers import R_DST

from sklearn.metrics import accuracy_score

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
    
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11], p_norm=0.8,
                 percentiles=[5, 10], n_jobs=1, random_state=None,
                 class_weight=None, fit_intercept=True,
                 alphas=np.logspace(-4,4,10)):

        self.classifier = make_pipeline(
            StandardScaler(with_mean=True),
            RidgeClassifierCV(
                alphas=alphas, class_weight=class_weight, 
                fit_intercept=fit_intercept
            )
        )
        self.transformer = R_DST(
            n_shapelets=n_shapelets, shapelet_sizes=shapelet_sizes, 
            p_norm=p_norm, percentiles=percentiles,
            n_jobs=n_jobs, random_state=random_state
        )
        
        
    def __repr__(self):
        return self.transformer.__repr__() + " " + self.classifier.__repr__()
    
    def _more_tags(self):
        return ["R_DST"]
    
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
        X = self.transformer.transform(X)
        return self.classifier.predict(X)
    
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

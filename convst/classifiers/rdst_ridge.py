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
    
    
    Attributes
    ----------

    Parameters
    ----------
    n_shapelets : TYPE, optional
        DESCRIPTION. The default is 10000.
    shapelet_sizes : TYPE, optional
        DESCRIPTION. The default is [11].
    p_norm : TYPE, optional
        DESCRIPTION. The default is 0.8.
    percentiles : TYPE, optional
        DESCRIPTION. The default is [5, 10].
    n_jobs : TYPE, optional
        DESCRIPTION. The default is -1.
    random_state : TYPE, optional
        DESCRIPTION. The default is None.
    class_weight : TYPE, optional
        DESCRIPTION. The default is None.
    fit_intercept : TYPE, optional
        DESCRIPTION. The default is True.
    alphas : TYPE, optional
        DESCRIPTION. The default is np.logspace(-4,4,10).

    Returns
    -------
    None.

    """
    
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11], p_norm=0.8,
                 percentiles=[5, 10], n_jobs=-1, random_state=None,
                 class_weight=None, fit_intercept=True,
                 alphas=np.logspace(-4,4,10)):

        self.classifier = make_pipeline(
            StandardScaler(with_mean=False),
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
    
    def fit(self, X, y):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.transformer = self.transformer.fit(X, y)
        self.classifier = self.classifier.fit(self.transformer.transform(X), y)
        return self
        
    def predict(self, X):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        check_is_fitted(self, ['classifier'])
        X = self.transformer.transform(X)
        return self.classifier.predict(X)
    
    def score(self, X, y):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        preds = self.predict(X)
        return accuracy_score(y, preds)


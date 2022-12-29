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
         must be a positive float.
         The default is np.logspace(-4,4,20).
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

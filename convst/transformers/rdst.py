#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 09:20:49 2022

@author: Antoine Guillaume
"""
import numpy as np
import warnings

from sklearn.utils import resample
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from convst.utils.checks_utils import (
    check_array_3D, check_array_1D, check_is_numeric, 
    check_is_boolean, check_n_jobs
)
from convst.transformers._commons import manhattan, euclidean, squared_euclidean

from numba import set_num_threads

STR_MUTLIVARIATE = 'multivariate'
STR_UNIVARIATE = 'univariate'
STR_UNIVARIATE_VARIABLE = 'univariate_variable'
STR_MULTIVARIATE_VARIABLE = 'multivariate_variable'

class R_DST(BaseEstimator, TransformerMixin):
    """
    Base class for RDST transformer. Depending on the parameters and of the
    type of data (i.e. multivariate, variable length, etc.) given during the
    fit method, it will call the adapted transformer.

    For more information on the transformer and the effect of the different 
    parameters on the transformation and shapelet extraction process, please
    refer to [1]_ and [2]_

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
        and time series. The default is 'euclidean'.
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
        extraction. The default is 1.0.
    n_shapelets : int, optional
        The maximum number of shapelet to be sampled. The default is 10_000.
    shapelet_lengths : array, optional
        The set of possible length for shapelets. The values can be integers
        to specify an absolute length, or a float, to specify a length relative 
        to the input time series length. The default is [11].
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

    Attributes
    -------
    transformer_ : object
        The transformer that have been selected based on the parameters and the
        type of data. This is the object used to transform the input data.


    .. [1] Antoine Guillaume et al, "Random Dilated Shapelet Transform: A new approach of time series shapelets" (2022)
    .. [2] Antoine Guillaume, "Time series classification with shapelets: Application to predictive maintenance on event logs" (2023)

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
        random_state=None,
        max_channels=None,
        min_len=None,
        n_jobs=1
    ):
        self.transform_type = self._validate_transform_type(transform_type)
        self.phase_invariance = check_is_boolean(phase_invariance)
        self.distance = self._validate_distances(distance)
        self.alpha = check_is_numeric(alpha)
        self.normalize_output = check_is_boolean(normalize_output)
        self.n_samples = check_is_numeric(n_samples) if n_samples is not None else n_samples
        self.n_shapelets = int(check_is_numeric(n_shapelets))
        self.shapelet_lengths = check_array_1D(shapelet_lengths)
        if shapelet_lengths_bounds is None:
            self.shapelet_lengths_bounds = None
        elif len(shapelet_lengths_bounds)==2:
            self.shapelet_lengths_bounds = shapelet_lengths_bounds
        else:
            raise ValueError('Shapelets lengths bounds should be a 1D array with 2 values')
        self.lengths_bounds_reduction=check_is_numeric(lengths_bounds_reduction)
        if self.lengths_bounds_reduction>=1:
            raise ValueError('lengths_bounds_reduction parameter should be in range [0,1[')
        self.prime_dilations = check_is_boolean(prime_dilations)
        self.proba_norm = check_is_numeric(proba_norm)
        self.percentiles = self._validate_percentiles(percentiles)
        self.random_state = check_random_state(random_state)
        if isinstance(n_jobs, bool):
            self.n_jobs=n_jobs
        else:
            self.n_jobs=check_n_jobs(n_jobs)
            set_num_threads(self.n_jobs)
        self.max_channels=max_channels
        self.min_len=min_len
    
    def _set_lengths(self):
        if self.shapelet_lengths_bounds is None:
            if self.shapelet_lengths.dtype == float:
                return np.floor(self.min_len*self.shapelet_lengths)
            else:
                return self.shapelet_lengths
        else:
            b0 = self.shapelet_lengths_bounds[0]
            b1 = self.shapelet_lengths_bounds[1]
            
            if isinstance(b0, float):
                b0 = int(b0*self.min_len)
            min_l = max(5,b0)
            if isinstance(b1, float):
                b1 = int(b1*self.min_len)
            max_l = max(6,max(b0+1,b1+1))
            #6 to ensure range 5,6 -> 5
            lengths = np.asarray(list(range(min_l, max_l)))
            if lengths.shape[0]>3:
                n_remove = int(lengths.shape[0]*self.lengths_bounds_reduction)
                if n_remove > 0:
                    step = lengths.shape[0]//n_remove
                    lengths = lengths[::step]
            return lengths
        
    def fit(self, X, y):
        """
        Fit method. Random shapelets are generated using the parameters
        supplied during initialisation. Then, the class attributes are filled 
        with the result of this random initialisation.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Input time series.
            
        y : array, shape=(n_samples)
            Class of the input time series.

        """
        self._set_fit_transform(X)
        if self.transform_type in [STR_MULTIVARIATE_VARIABLE, STR_UNIVARIATE_VARIABLE]:
            X, X_len = self._format_uneven_timestamps(X)
            X = check_array_3D(X, is_univariate=False).astype(np.float64)
            if self.min_len is None:
                self.min_len = X_len.min()
        else:
            X = check_array_3D(X, is_univariate=False).astype(np.float64)
            self.min_len = X.shape[2]
        
        if self.n_samples is None:
            pass
        elif self.n_samples < 1.0:
            id_X = resample(np.arange(X.shape[0]), replace=False, n_samples=int(X.shape[0]*self.n_samples), stratify=y)
            X = X[id_X]
            y = y[id_X]
        elif self.n_samples > 1.0:
            id_X = resample(np.arange(X.shape[0]), replace=True, n_samples=int(X.shape[0]*self.n_samples), stratify=y)
            X = X[id_X]
            y = y[id_X]
        
        n_samples, n_features, _ = X.shape
        
        if self.max_channels is None:
            self.max_channels = n_features
        
        
        self.shapelet_lengths = self._set_lengths()
        
        shapelet_lengths, seed = self._check_params(self.min_len)
        # Generate the shapelets
        if self.transform_type == STR_UNIVARIATE_VARIABLE:
            self.shapelets_ = self.fitter(
                X, y, self.n_shapelets, shapelet_lengths, seed, self.proba_norm,
                self.percentiles[0], self.percentiles[1], self.alpha,
                self._get_distance_function(), self.phase_invariance,
                self.min_len, X_len, self.prime_dilations
            )
        elif self.transform_type == STR_MULTIVARIATE_VARIABLE:
            self.shapelets_ = self.fitter(
                X, y, self.n_shapelets, shapelet_lengths, seed, self.proba_norm,
                self.percentiles[0], self.percentiles[1], self.alpha,
                self._get_distance_function(), self.phase_invariance,
                self.max_channels, self.min_len, X_len, self.prime_dilations
            )
        elif self.transform_type == STR_MUTLIVARIATE:
            self.shapelets_ = self.fitter(
                X, y, self.n_shapelets, shapelet_lengths, seed, self.proba_norm,
                self.percentiles[0], self.percentiles[1], self.alpha, 
                self._get_distance_function(), self.phase_invariance, 
                self.max_channels, self.prime_dilations
            )
        elif self.transform_type == STR_UNIVARIATE:
            self.shapelets_ = self.fitter(
                X, y, self.n_shapelets, shapelet_lengths, seed, self.proba_norm,
                self.percentiles[0], self.percentiles[1], self.alpha, 
                self._get_distance_function(), self.phase_invariance, self.prime_dilations
            )
        else:
            raise ValueError('Unknown value for transform type parameter')
        
        return self


    def transform(self, X):
        """
        Transform the input time series using previously fitted shapelets. 
        We compute a distance vector between each shapelet and each time series
        and extract the min, argmin, and shapelet occurence features based on
        the lambda threshold of each shapelet.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Input time series.

        Returns
        -------
        X : array, shape=(n_samples, 3*n_shapelets)
            Transformed input time series.

        """
        
        check_is_fitted(self, ['shapelets_'])
        if self.transform_type in [STR_MULTIVARIATE_VARIABLE, STR_UNIVARIATE_VARIABLE]:
            X, X_len = self._format_uneven_timestamps(X)
            X = check_array_3D(X).astype(np.float64)
            X_new = self.transformer(
                X, self.shapelets_ , self._get_distance_function(),
                self.phase_invariance, X_len
            )
        else:
            X = check_array_3D(X).astype(np.float64)
            X_new = self.transformer(
                X, self.shapelets_, self._get_distance_function(),
                self.phase_invariance
            )
        return X_new
    
    def _auto_class(self, X):
        """
        Using the input time series data, find the type of transformation to 
        use. Either univariate or multivariate, and either same length or 
        variable length.

        Parameters
        ----------
        X : array, shape=(n_samples, n_featrues, n_timestamps)
            The input time series data. For variable length time series, it 
            can either be a list of 2D array, or a numpy array with object
            dtype or a python list of arrays.

        Returns
        -------
        str
            The type of transformation to use.

        """
        #[STR_UNIVARIATE,STR_MUTLIVARIATE,STR_UNIVARIATE,STR_MULTIVARIATE_VARIABLE]
        if isinstance(X, list) or X.dtype == np.object_:
            if len(X[0]) > 1:
                return STR_MULTIVARIATE_VARIABLE
            else:
                return STR_UNIVARIATE_VARIABLE
        elif X.dtype == np.int_ or X.dtype == np.float_:
            #Even length
            X = check_array_3D(X)
            if X.shape[1] > 1:
                return STR_MUTLIVARIATE
            else:
                return STR_UNIVARIATE
        
        
    def _set_fit_transform(self, X):
        """
        Based on the transformation type, either specified by the user or found
        by the _auto_class method, initialize the fitter and transformer 
        attributes.

        Parameters
        ----------
        X : array, shape=(n_samples, n_featrues, n_timestamps)
            The input time series data. For variable length time series, it 
            can either be a list of 2D array, or a numpy array with object
            dtype.
            
        Returns
        -------
        None.

        """
        if self.transform_type == 'auto':
            _type = self._auto_class(X)
            self.transform_type = _type
        else:
            _type = self.transform_type
            
        if _type == STR_UNIVARIATE:
            from convst.transformers._univariate_same_length import (
                U_SL_apply_all_shapelets, U_SL_generate_shapelet
            )
            self.fitter = U_SL_generate_shapelet
            self.transformer = U_SL_apply_all_shapelets
            
        elif _type == STR_MUTLIVARIATE:
            from convst.transformers._multivariate_same_length import (
                M_SL_apply_all_shapelets, M_SL_generate_shapelet
            )
            self.fitter = M_SL_generate_shapelet
            self.transformer = M_SL_apply_all_shapelets
            
        elif _type == STR_UNIVARIATE_VARIABLE:
            from convst.transformers._univariate_variable_length import (
                U_VL_apply_all_shapelets, U_VL_generate_shapelet
            )
            self.fitter = U_VL_generate_shapelet
            self.transformer = U_VL_apply_all_shapelets
            
        elif _type == STR_MULTIVARIATE_VARIABLE:
            from convst.transformers._multivariate_variable_length import (
                M_VL_apply_all_shapelets, M_VL_generate_shapelet
            )
            self.fitter = M_VL_generate_shapelet
            self.transformer = M_VL_apply_all_shapelets
        
        else:
            raise ValueError('Unknwon transform type parameter')
        
    
    def _get_distance_function(self):
        """
        Based on the distance parameter, return the distance function to be 
        used during the shapelet generation and transform.

        Raises
        ------
        ValueError
            If the value of the distance parameter is not in ['euclidean', 
            'squared','manhattan'], raise a ValueError.

        Returns
        -------
        function
            Return the numba function based on the distance parameter.
            
        """
        if self.distance == 'euclidean':
            return euclidean
        if self.distance == 'squared':
            return squared_euclidean
        if self.distance == 'manhattan':
            return manhattan
        raise ValueError('Wrong distance parameter value, got {}'.format(self.distance))
    
    def _format_uneven_timestamps(self, X):
        """
        Given a set of variable length time series, create a 3D numpy array 
        of float dtype to be used in the numba function. This will create an 
        array with n_timestamps equal to the maximum length in X.

        Parameters
        ----------
        X : array, shape=(n_samples, n_featrues, n_timestamps)
             The input time series data. For variable length time series, it 
             can either be a list of 2D array, or a numpy array with object
             dtype.

        Raises
        ------
        ValueError
            If the number of feature is different between samples a ValueError
            is raised.

        Returns
        -------
        X_new : array, shape=(n_samples, n_featrues, n_timestamps)
             The input time series data. For variable length time series, it 
             can either be a list of 2D array, or a numpy array with object
             dtype.
        lengths : array, shape=(n_samples)
             The true length of each input time series.

        """
        n_ft = np.zeros(len(X),dtype=np.int64)
        lengths = np.zeros(len(X),dtype=np.int64)
        for i in range(len(X)):
            n_ft[i] = X[i].shape[0]
            lengths[i] = X[i].shape[1]
        
        if np.all(n_ft == n_ft[0]):      
            X_new = np.zeros((len(X), n_ft[0], lengths.max()))
            for i in range(len(X)):
                X_new[i, :, :lengths[i]] = X[i]
            return X_new, lengths
        else:
            raise ValueError("Samples got different number of features")
    

    def _check_params(self, n_timestamps):
        """
        Check the validity of the input parameters, notably that shapelet 
        length do not exceed time series lengths, and that we have at least 3
        timestamp in all series.

        Parameters
        ----------
        n_timestamps : int
            Number of timestamps of input time series, if the input is of 
            variable length, it must be the minimum number of timestamps.

        Raises
        ------
        TypeError
            If the number of shapelet is not an integer, or the shapelet lengths
            are not a collection.
        ValueError
            If the size of the shapelets is not valid or the time series have
            less than 3 timestamps.

        Returns
        -------
        shapelet_lengths : array
            The possible size of the shapelets.
        seed : int
            The random seed to be passed to the shapelet generator.

        """
        if not isinstance(self.n_shapelets, (int, np.integer)):
            raise TypeError("'n_shapelets' must be an integer (got {})."
                            .format(self.n_shapelets))

        if not isinstance(self.shapelet_lengths, (list, tuple, np.ndarray)):
            raise TypeError("'shapelet_lengths' must be a list, a tuple or "
                            "an array (got {}).".format(self.shapelet_lengths))
        
        shapelet_lengths = check_array_1D(self.shapelet_lengths).astype(np.int64)
        if not np.all(1 < shapelet_lengths):
            raise ValueError("All the values in 'shapelet_lengths' must be "
                             "greater than 1, but got {} ({} < 1)."
                             .format(shapelet_lengths, shapelet_lengths.min()))
            
        if not np.all(shapelet_lengths <= n_timestamps):
            if n_timestamps < 3:
                raise ValueError('Input data goint {} timestamps, at least 4 are requiered. Input format should be (n_samples, n_features, n_timestamps)'.format(n_timestamps))
            else:
                warnings.warn("All the values in 'shapelet_lengths' must be lower than or equal to 'n_timestamps' (got {} > {}). Changed shapelet size to {}".format(shapelet_lengths.max(), n_timestamps, n_timestamps//2))
                shapelet_lengths[shapelet_lengths > n_timestamps] = n_timestamps//2


        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

        return shapelet_lengths, seed    
    
    def _validate_transform_type(self, transform_type):
        transform_type = transform_type.lower()
        valid = ['auto',STR_UNIVARIATE,STR_MUTLIVARIATE,STR_UNIVARIATE,STR_MULTIVARIATE_VARIABLE]
        if transform_type not in valid:
            raise ValueError('Wrong transform_type parameter value, got {}, valid ones are {}'.format(transform_type, valid))
        return transform_type
    
    def _validate_percentiles(self, percentiles):
        percentiles = check_array_1D(percentiles)
        if percentiles.dtype == int or percentiles.dtype == float :
            if percentiles[0] <= percentiles[1]:
                return percentiles
        raise ValueError('Wrong percetniles parameter value, got {}, expected a numerical array of size 2'.format(percentiles))
    
    def _validate_distances(self, distance_str):
        distance_str = distance_str.lower()
        valid = ['euclidean','squared','manhattan']
        if distance_str not in valid:
            raise ValueError('Wrong distance parameter value, got {}, valid ones are {}'.format(distance_str, valid))
        return distance_str
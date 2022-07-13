# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:17:07 2021

@author: a694772
"""

import numpy as np
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from convst.utils.checks_utils import check_array_3D, check_array_1D
from convst.utils.shapelets_utils import generate_strides_1D

from numba import set_num_threads
from numba import njit, prange
from numba.core.config import NUMBA_DEFAULT_NUM_THREADS
from convst.utils.checks_utils import check_n_jobs

from matplotlib import pyplot as plt

from convst.transformers.dummies.catch22 import (
    DN_HistogramMode_5,
    DN_HistogramMode_10,
    SB_BinaryStats_diff_longstretch0,
    DN_OutlierInclude_p_001_mdrmd,
    DN_OutlierInclude_n_001_mdrmd,
    SP_Summaries_welch_rect_area_5_1,
    SP_Summaries_welch_rect_centroid,
    FC_LocalSimple_mean3_stderr,
    CO_trev_1_num,
    CO_HistogramAMI_even_2_5,
    MD_hrv_classic_pnn40,
    SB_BinaryStats_mean_longstretch1,
    SB_MotifThree_quantile_hh,
    SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
    SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
    PD_PeriodicityWang_th0_01,
)

@njit(fastmath=True, cache=True, error_model='numpy')
def compute_shapelet_dist_vector(x, values, length, dilation, normalize):
    """
    Compute a shapelet distance vector from an univariate time series 
    and a dilated shapelet. Shapelet should be already normalized if normalizing
    the distance

    Parameters
    ----------
    x : array, shape=(n_timestamps)
        An input time series
    values : array, shape=(max_shapelet_length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    dilation : int
        Dilation of the shapelet
    normalize : float
        Boolean converted as float (to avoid if statement) indicating
        wheter or not the distance will be z-normalized

    Returns
    -------
    x_conv : array, shape=(n_timestamps - (length-1) * dilation)
        The resulting distance vector

    """
    c = generate_strides_1D(x, length, dilation)
    x_conv = np.empty(c.shape[0])
    for i in range(x_conv.shape[0]):
        s = 0
        mean = c[i].mean()*normalize
        std = (c[i].std()+1e-8)*normalize
        x0 = (c[i] - mean)/(std + 1.0-normalize)
        for j in prange(length):
            s += abs(x0[j] - values[j])
        x_conv[i] = s
    return x_conv


@njit(cache=True, parallel=True)
def _init_random_shapelet_params(n_shapelets, shapelet_sizes, n_timestamps, p_norm):
    """
    Initialize the parameters of the shapelets.    

    Parameters
    ----------
    n_shapelets : int
        Number of shapelet to initialize
    shapelet_sizes : array, shape=()
        Set of possible length for the shapelets
    n_timestamps : int
        Number of timestamps in the input data
    p_norm : float
        A value in the range [0,1] indicating the chance for each
        shapelet to use z-normalized distance

    Returns
    -------
    values : array, shape=(n_shapelet, max(shapelet_sizes))
        An initialized (empty) value array for each shapelet
    lengths : array, shape=(n_shapelet)
        The randomly initialized length of each shapelet
    dilations : array, shape=(n_shapelet)
        The randomly initialized dilation of each shapelet
    threshold : array, shape=(n_shapelet)
        An initialized (empty) value array for each shapelet
    normalize : array, shape=(n_shapelet)
        The randomly initialized normalization indicator of each shapelet

    """
    # Lengths of the shapelets
    lengths = np.random.choice(
        shapelet_sizes, size=n_shapelets).astype(np.int64)

    # Dilations
    upper_bounds = np.log2(np.floor_divide(n_timestamps - 1, lengths - 1))
    powers = np.empty(n_shapelets)
    for i in prange(n_shapelets):
        powers[i] = np.random.uniform(0, upper_bounds[i])
    dilations = np.floor(np.power(2, powers)).astype(np.int64)

    # Init threshold array
    threshold = np.zeros(n_shapelets, dtype=np.float64)

    # Init values array
    values = np.zeros((n_shapelets, np.int64(np.max(shapelet_sizes))),
                      dtype=np.float64)

    # Is shapelet using z-normalization ?
    normalize = np.random.random(size=n_shapelets)
    normalize = (normalize < p_norm).astype(np.float64)

    return values, lengths, dilations, threshold, normalize


@njit(cache=True, fastmath=True, error_model='numpy')
def _get_subsequence(X, i_start, l, d, normalize):
    """
    Given a set of length and dilation, fetch a subsequence from an input 
    time series given a starting index.

    Parameters
    ----------
    X : array, shape=(n_timestamps)
        Input time series.
    i_start : int
        A starting index between [0, n_timestamps - (l-1)*d]
    l : int
        Length parameter.
    d : int
        Dilation parameter.
    normalize : float
        Boolean converted as float (to avoid if statement) indicating
        wheter or not the distance will be z-normalized

    Returns
    -------
    v : array, shape=(l)
        The resulting subsequence.

    """
    v = np.empty(l, dtype=np.float64)
    _idx = i_start
    _sum = 0
    _sum2 = 0
    for j in prange(l):
        v[j] = X[_idx]
        _sum += X[_idx]
        _sum2 += X[_idx]**2
        _idx += d
    #0 if normalize, seems faster than adding a if statement
    mean = (_sum/l)*normalize
    std = (np.sqrt((_sum2/l) - mean**2)+1e-8)*normalize
    # divided by 1 if non normalized
    v = (v - mean)/(std + 1-normalize)
    return v


@njit(cache=True, parallel=True)
def generate_shapelet(X, y, n_shapelets, shapelet_sizes, seed, p_norm, p_min, p_max):
    """
    Given a time series dataset and parameters of the method, generate the
    set of random shapelet that will be used in the rest of the algorithm.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Time series dataset
    y : array, shape=(n_samples)
        Class of each input time series
    n_shapelets : int
        Number of shapelet to generate
    shapelet_sizes : array, shape=()
        An array of possible shapelet length.
    seed : int
        Random seed generator for numpy
    p_norm : float
        Probability of each shapelet to use z-normalized distance
    p_min : float
        Lower bound for the percentile during the choice of threshold
    p_max : float
        Upper bound for the percentile during the choice of threshold

    Returns
    -------
    set of arrays
        Return the values, lengths, dilations, thresholds and normalization
        indicators as array with first dimension of size (n_shapelets)
    """
    n_samples, n_features, n_timestamps = X.shape

    # Fix the random see
    np.random.seed(seed)

    values, lengths, dilations, threshold, normalize = _init_random_shapelet_params(
        n_shapelets, shapelet_sizes, n_timestamps, p_norm
    )

    samples_pool = np.arange(X.shape[0]).astype(np.int64)
    np.random.shuffle(samples_pool)
    # For Values, draw from random uniform (0,n_samples*(n_ts-(l-1)*d))
    # for each l,d combinations. Then take by index the values instead
    # of generating strides.
    for i in prange(n_shapelets):
        id_sample = samples_pool[i % X.shape[0]]
        index = np.int64(np.random.choice(
            n_timestamps - (lengths[i]-1)*dilations[i]
        ))
        
        v = _get_subsequence(
            X[id_sample, 0], index, lengths[i], dilations[i], normalize[i]
        )

        values[i, :lengths[i]] = v

        id_test = np.random.choice(np.where(y == y[id_sample])[0])

        x_dist = compute_shapelet_dist_vector(
            X[id_test, 0], values[i], lengths[i], dilations[i], normalize[i]
        )
        threshold[i] = np.random.uniform(
            np.percentile(x_dist, p_min), np.percentile(x_dist, p_max)
        )
        
    return values, lengths, dilations, threshold, normalize.astype(np.int64)


@njit(cache=False, parallel=True, fastmath=True, error_model='numpy')
def apply_all_shapelets(X, values, lengths, dilations, threshold, normalize):
    """
    Apply a set of generated shapelet using the parameter arrays previously 
    generated to a set of time series.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Input time series
    values : array, shape=(n_shapelets, max(shapelet_sizes))
        Values of the shapelets. If the shapelet use z-normalized distance,
        those values are already z-normalized by the shapelet 
        initialization step.
    lengths : array, shape=(n_shapelets)
        Length parameter of the shapelets
    dilations : array, shape=(n_shapelets)
        Dilation parameter of the shapelets
    threshold : array, shape=(n_shapelets)
        Threshold parameter of the shapelets
    normalize : array, shape=(n_shapelets)
        Normalization indicatorr of the shapelets

    Returns
    -------
    X_new : array, shape=(n_samples, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.

    """
    n_samples, n_ft, n_timestamps = X.shape
    n_shapelets = len(lengths)
    n_features = 17

    unique_lengths = np.unique(lengths)
    unique_dilations = np.unique(dilations)

    X_new = np.empty((n_samples, n_features * n_shapelets), dtype=np.float64)
    for index_l in range(unique_lengths.shape[0]):
        l = unique_lengths[index_l]
        for index_d in prange(unique_dilations.shape[0]):
            d = unique_dilations[index_d]

            ix_shapelets = np.where((lengths == l) & (dilations == d))[0]
            d_shape = n_timestamps - (l-1)*d

            if len(ix_shapelets) > 0:
                for i in prange(n_samples):
                    strides = generate_strides_1D(X[i, 0], l, d)
                    X_sample = np.empty((2, d_shape, l), dtype=np.float64)
                    X_sample[0] = strides
                    X_sample[1] = strides
                    for j in range(d_shape):
                        X_sample[1, j] = (X_sample[1, j] - np.mean(X_sample[1, j]))/(
                            np.std(X_sample[1, j])+1e-8
                        )

                    for j in prange(len(ix_shapelets)):
                        i_shp = ix_shapelets[j]
                        X_new[i, (n_features * i_shp):(n_features * i_shp + n_features)] = apply_one_shapelet_one_sample(
                            X_sample[normalize[i_shp]
                                     ], values[i_shp], threshold[i_shp]
                        )
    return X_new


@njit(fastmath=True, cache=False)
def apply_one_shapelet_one_sample(x, values, threshold):
    """
    Extract the three features from the distance between a shapelet and the 
    strides of an input time series generated by the length and dilation 
    parameter of the shapelet. If normalization should be used, all strides
    of should be z-normalized independently.

    Parameters
    ----------
    x : array, shape=(n_timestamps - (length-1)*dilation, length)
        Strides of an input time series
    values : array, shape=(max(shapelet_sizes))
        Values of the shapelet
    threshold : float
        The threshold to compute the shapelet occurence feature.
    
    Returns
    -------
    _min : float
        The minimum euclidean distance between the shapelet and the input time
        series
    float
        The location of the minimum euclidean distance divided by the length 
        of the distance vector (i.e scaled between [0,1]).        
    float
        The number of points in the distance vector inferior to the threshold 
        divided by the length of the distance vector (i.e scaled between [0,1])

    """
    n_candidates, length = x.shape
    dist_vect = np.zeros(n_candidates)
    #For each step of the moving window in the shapelet distance
    for i in range(n_candidates):
        _dist = 0
        #For each value of the shapelet
        for j in prange(length):
            _dist += abs(x[i, j] - values[j])

        dist_vect[i] = _dist

    outputs = np.zeros(17)
    outputs[0] = np.min(dist_vect)
    outputs[1] = np.float64(np.argmin(dist_vect))
    outputs[2] = np.float64(np.sum(dist_vect<threshold))
    outputs[3] = DN_HistogramMode_5(dist_vect)
    outputs[4] = DN_HistogramMode_10(dist_vect)
    outputs[5] = SB_BinaryStats_diff_longstretch0(dist_vect)
    outputs[6] = DN_OutlierInclude_p_001_mdrmd(dist_vect)
    outputs[7] = DN_OutlierInclude_n_001_mdrmd(dist_vect)
    outputs[8] = SP_Summaries_welch_rect_area_5_1(dist_vect)
    outputs[9] = SP_Summaries_welch_rect_centroid(dist_vect)
    outputs[10] = FC_LocalSimple_mean3_stderr(dist_vect)
    outputs[11] = CO_trev_1_num(dist_vect)
    outputs[12] = CO_HistogramAMI_even_2_5(dist_vect)
    outputs[13] = MD_hrv_classic_pnn40(dist_vect)
    outputs[14] = SB_BinaryStats_mean_longstretch1(dist_vect)
    outputs[15] = SB_MotifThree_quantile_hh(dist_vect)
    #outputs[16] = SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(dist_vect)
    #outputs[17] = SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(dist_vect)
    outputs[16] = PD_PeriodicityWang_th0_01(dist_vect)
    return outputs


class R_DST_22(BaseEstimator, TransformerMixin):
    """
    Implementation of univariate Random Dilated Shapelet Transform (RDST).
    For details and explanation on the algorithm, users are referred to [1]_:
        
    Parameters
    ----------
    n_shapelets : int, optional
        Number of shapelets to generate. The default is 10000.
    shapelet_sizes : array
        An array of int which indicate the possible absolute shapelet sizes.
        Or an array of float which will give shapelet sizes relative to input length.
        The default is [11]
    p_norm : float
        A float between 0 and 1 indicating the proportion of shapelets that
        will use a z-normalized distance. The default is 0.8.
    percentiles : array, shape=(2)
        The two percentiles (between 0 and 100) between which the value of the
        threshold will be sampled during shapelet generation. 
        The default is [5,10].
    random_state : int, optional
        Value of the random state for all random number generation.
        The default is None.
    n_jobs : int, optional
        Number of thread used by numba for the computational heavy part
        of the algortihm. The default is -1 (i.e all available cores).
        
    Attributes
    ----------
    values_ : array, shape=(n_shapelets)
        The values of each shapelets will be stored in this
        array after calling the fit method.
    length_ : array, shape=(n_shapelets)
        The length parameter of each shapelets will be stored in this
        array after calling the fit method.
    dilation_ : array, shape=(n_shapelets)
        The dilation parameter of each shapelets will be stored in this
        array after calling the fit method.
    normalize_ : array, shape=(n_shapelets)
        The normalize parameter of each shapelets will be stored in this
        array after calling the fit method.
    threshold_ : array, shape=(n_shapelets)
        The threshold parameter of each shapelets will be stored in this
        array after calling the fit method.
        

    .. [1] Antoine Guillaume et al, "Random Dilated Shapelet Transform: A new approach of time series shapelets" (2022)
    """
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11], n_jobs=1,
                 p_norm=0.8, percentiles=[5, 10], random_state=None):
        self.n_shapelets = n_shapelets
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.p_norm = p_norm
        self.percentiles = percentiles
        self.n_jobs = check_n_jobs(n_jobs)
        set_num_threads(n_jobs)
            
            
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
        X = check_array_3D(X, is_univariate=True).astype(np.float64)
        n_samples, n_features, n_timestamps = X.shape
        if self.shapelet_sizes.dtype == float:
            self.shapelet_sizes = np.floor(n_timestamps*self.shapelet_sizes)
        
        shapelet_sizes, seed = self._check_params(n_timestamps)
        # Generate the shapelets

        values, lengths, dilations, threshold, normalize = generate_shapelet(
            X, y, self.n_shapelets, shapelet_sizes, seed, self.p_norm,
            self.percentiles[0], self.percentiles[1]
        )
        self.values_ = values
        self.length_ = lengths
        self.dilation_ = dilations
        self.normalize_ = normalize
        self.threshold_ = threshold

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
        X = check_array_3D(X, is_univariate=True).astype(np.float64)
        check_is_fitted(self, ['values_', 'length_',
                        'dilation_', 'threshold_', 'normalize_'])
        X_new = apply_all_shapelets(
            X, self.values_, self.length_, self.dilation_, self.threshold_, self.normalize_
        )
        return X_new

    def _check_params(self, n_timestamps):
        if not isinstance(self.n_shapelets, (int, np.integer)):
            raise TypeError("'n_shapelets' must be an integer (got {})."
                            .format(self.n_shapelets))

        if not isinstance(self.shapelet_sizes, (list, tuple, np.ndarray)):
            raise TypeError("'shapelet_sizes' must be a list, a tuple or "
                            "an array (got {}).".format(self.shapelet_sizes))
        shapelet_sizes = check_array_1D(self.shapelet_sizes).astype(np.int64)
        
        if not np.all(1 <= shapelet_sizes):
            raise ValueError("All the values in 'shapelet_sizes' must be "
                             "greater than or equal to 1 ({} < 1)."
                             .format(shapelet_sizes.min()))
            
        if not np.all(shapelet_sizes <= n_timestamps):
            raise ValueError("All the values in 'shapelet_sizes' must be lower "
                             "than or equal to 'n_timestamps' (got {} > {})."
                             .format(shapelet_sizes.max(), n_timestamps))

        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

        return shapelet_sizes, seed

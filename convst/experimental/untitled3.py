# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:41:56 2022

@author: a694772
"""

import numpy as np
from numpy.random import choice

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from convst.utils.numba_utils import (
    arange, _abs, empty, zeros, ones, log2, floor, sum_array, _sum, mean,
    mean_array, _min, _max, argmin, std, std_array, floor_divide,
    sqrt, power, seed, uniform, arange3, arange2, cumsum
)
from convst.utils.checks_utils import check_array_3D, check_array_1D
from convst.utils.shapelets_utils import generate_strides_1D

from numba import set_num_threads, njit, prange, float64, int64
from numba.core.config import NUMBA_DEFAULT_NUM_THREADS

@njit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _gini_split(y_node, y_left, y_right, class_weights):
    n_targets = class_weights.shape[0]
    n_classes_node = zeros(n_targets)
    n_classes_left = zeros(n_targets)
    n_classes_right = zeros(n_targets)
    total_sum_node = 0
    total_sum_left = 0
    total_sum_right = 0
    for k in prange(n_targets):
        n_classes_node[k] = _sum(y_node == k) * class_weights[k]
        n_classes_left[k] = _sum(y_left == k) * class_weights[k]
        n_classes_right[k] = _sum(y_right == k) * class_weights[k]
        total_sum_node += n_classes_node[k]
        total_sum_left += n_classes_left[k]
        total_sum_right += n_classes_right[k]
    
    gini_node = 1
    gini_left = 1
    gini_right = 1
    for k in prange(n_targets):
        gini_node -= (n_classes_node[k]/total_sum_node)**2
        gini_left -= (n_classes_left[k]/total_sum_left)**2
        gini_right -= (n_classes_right[k]/total_sum_right)**2
    prop_left = (total_sum_left/total_sum_node)
    prop_right = (total_sum_right/total_sum_node)
    gain = gini_node - (prop_left*gini_left) - (prop_right*gini_right)
    return gain

@njit(fastmath=True, cache=True, error_model='numpy', nogil=True)
def compute_distance_vector(x, values, length, dilation, normalize):
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
    x_conv = empty(c.shape[0])
    for i in arange(x_conv.shape[0]):
        s = 0
        _mean = mean(c[i])*normalize
        _std = (std(c[i])+1e-8)*normalize
        x0 = (c[i] - _mean)/(_std + 1.0-normalize)
        for j in arange(length):
            s += _abs(x0[j] - values[j])
        x_conv[i] = s
    return x_conv

@njit(cache=True, fastmath=True, error_model='numpy', nogil=True)
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
    v = empty(l, dtype=float64)
    _idx = i_start
    __sum = 0
    __sum2 = 0
    for j in range(l):
        v[j] = X[_idx]
        __sum += X[_idx]
        __sum2 += X[_idx]**2
        _idx += d
    #0 if normalize, seems faster than adding a if statement
    _mean = (__sum/l)*normalize
    _std = (np.sqrt((__sum2/l) - _mean**2)+1e-8)*normalize
    # divided by 1 if non normalized
    v = (v - _mean)/(_std + 1-normalize)
    return v

@njit(cache=True, fastmath=True, nogil=True)
def _init_shapelet_params_array(size, l, X_size):
    X_new = empty((X_size, 3*size), dtype=float64)
    values = empty((size,l), dtype=float64)
    lengths = empty(size, dtype=int64)
    dilations = empty(size, dtype=int64)
    threshold = empty(size, dtype=float64)
    normalize = empty(size, dtype=float64)
    return X_new, values, lengths, dilations, threshold, normalize

@njit(cache=True, fastmath=True, nogil=True)
def _grow_params_array(
    X_new, values, lengths, dilations, threshold, normalize, base_size
):
    current_len = lengths.shape[0]
    n_x, n_v, n_l, n_d, n_t, n_n = _init_shapelet_params_array(
        current_len + base_size, values.shape[1], X_new.shape[0]
    )
    n_x[:,:3*current_len] = X_new
    n_v[:current_len] = values
    n_l[:current_len] = lengths
    n_d[:current_len] = dilations
    n_t[:current_len] = threshold
    n_n[:current_len] = normalize
    return n_x, n_v, n_l, n_d, n_t, n_n

@njit(cache=True, parallel=True, fastmath=True, error_model='numpy')
def generate_shapelet(X, y, l, _seed, p_norm, alpha_stop, n_steps, cw):
    """
    Given a time series dataset and parameters of the method, generate the
    set of random shapelet that will be used in the rest of the algorithm.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Time series dataset
    y : array, shape=(n_samples)
        Class of each input time series
    shapelet_sizes : array, shape=()
        An array of possible shapelet length.
    seed : int
        Random seed generator for numpy
    p_norm : float
        Probability of each shapelet to use z-normalized distance
    
    Returns
    -------
    set of arrays
        Return the values, lengths, dilations, thresholds and normalization
        indicators as array with first dimension of size (n_shapelets)
    """
    n_samples, n_features, n_timestamps = X.shape

    # Fix the random seed
    seed(_seed)
    base_size = int64(sqrt(n_samples*n_timestamps))
    max_x = log2(floor_divide(n_timestamps - 1, l - 1))
    
    max_d = int64(floor(power(2, max_x)))
    X_new, values, lengths, dilations, threshold, normalize = _init_shapelet_params_array(
        base_size, l, n_samples
    )
    
    W = ones((n_samples, n_timestamps, max_d), dtype=float64)
    for i in prange(max_d):
        for j in range(n_samples):
            W[j, n_timestamps - (l-1)*(i+1):, i] = 0
    W = W.reshape((-1, max_d))
    
    W_mean = mean(W)
    alpha_stop = alpha_stop * W_mean
    
    i_shp = 0
    #Can do for each D in parallel And monitor each W_mean by alpha for each d
    while W_mean > alpha_stop:
        if i_shp >= values.shape[0]:
            X_new, values, lengths, dilations, threshold, normalize = _grow_params_array(
                X_new, values, lengths, dilations, threshold, normalize, base_size
            )
        
        d = numba_choice_p(arange2(1,max_d+1), sum_array(W,0))
        norm = float64(np.random.random() < p_norm)
        n_timestamp_d = (n_timestamps - (l-1)*d)
        w_shape = n_samples * n_timestamp_d
        
        i_in = numba_choice_p(arange(w_shape), W[:w_shape, d-1])
        
        ix = i_in//n_timestamp_d
        ij = i_in%n_timestamp_d
        
        val = _get_subsequence(X[ix, 0], ij, l, d, norm)
        
        D = compute_distance_vectors(X, val, l, d, norm)
        
        th = np.random.uniform(
            np.percentile(D[ix], 2.5), np.percentile(D[ix], 7.5)
        )
        #Update weights
        for i in prange(n_samples):
            mask = D[i]<=th
            
            X_new[i, (i_shp*3)] = _min(D[i])
            X_new[i, (i_shp*3)+1] = argmin(D[i])
            X_new[i, (i_shp*3)+2] = _sum(mask)
            
            i_start = i*n_timestamps
            i_end = i_start+D.shape[1]
            W_i = W[i_start:i_end, d-1]
            W_i[mask] -= (D[i][mask]/th)
            W_i[W_i < 0] = 0
            W[i_start:i_end, d-1] = W_i
        
        values[i_shp] = val
        lengths[i_shp] = l
        dilations[i_shp] = d
        threshold[i_shp] = th
        normalize[i_shp] = norm
        
        i_shp += 1
        W_mean = mean(W)
        
    return X_new[:,:3*i_shp], values[:i_shp], lengths[:i_shp], dilations[:i_shp], threshold[:i_shp], normalize.astype(np.int64)[:i_shp]



@njit(cache=True, fastmath=True, nogil=True)
def numba_choice_p(arr, p):
    # Get cumulative weights
    wc = cumsum(p)
    # sum of weights to 1
    wc /= wc[-1]
    # Pick random weight value
    r = np.random.rand()
    # Get corresponding index
    idx = np.searchsorted(wc, r, side='right')
    return arr[idx]


@njit(cache=True, parallel=True, fastmath=True, error_model='numpy')
def compute_distance_vectors(X, values, length, dilation, normalize):
    out = empty((X.shape[0], X.shape[2] - (length-1) * dilation))
    for i in prange(X.shape[0]):
        out[i] = compute_distance_vector(
            X[i,0], values, length, dilation, normalize
        )
    return out
    

@njit(cache=True, parallel=True, fastmath=True, error_model='numpy')
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
    n_features = 3

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


@njit(fastmath=True, cache=True)
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

    _n_match = 0
    _min = 1e+100
    _argmin = 0

    #For each step of the moving window in the shapelet distance
    for i in range(n_candidates):
        _dist = 0
        #For each value of the shapelet
        for j in prange(length):
            _dist += abs(x[i, j] - values[j])

        if _dist < _min:
            _min = _dist
            _argmin = i

        if _dist <= threshold:
            _n_match += 1

    return _min, np.float64(_argmin), np.float64(_n_match)


class IR_DST2(BaseEstimator, ClassifierMixin):
    """
    
        
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
    def __init__(
            self, 
            shapelet_sizes=[11], 
            n_jobs=-1,
            alpha_stop=0.05,
            n_steps=20,
            p_norm=0.8,
            random_state=None,
            class_weight='balanced',
            classifier=None
        ):
        self.alpha_stop = alpha_stop
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.classifier = classifier
        self.class_weight = class_weight
        self.p_norm = p_norm
        self.n_steps = n_steps
        self.n_jobs = n_jobs
        if self.n_jobs == -1:
            set_num_threads(NUMBA_DEFAULT_NUM_THREADS)
        elif isinstance(self.n_jobs, int) and self.n_jobs > 0:
            set_num_threads(int(self.n_jobs))
        else:
            raise ValueError("n_jobs parameter should be a int superior to 0 or equal to -1 but got {}".format(self.n_jobs))
            
            
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
        n_classes = np.unique(y).shape[0]
        cw = np.ones(n_classes, dtype=np.float64)
        if self.class_weight == 'balanced':
            for i in range(n_classes):
                cw[i] = y.shape[0] / (n_classes * (y == i).sum())


        out, values, lengths, dilations, threshold, normalize = generate_shapelet(
            X, y, shapelet_sizes[0], seed, self.p_norm,
            self.alpha_stop, self.n_steps, cw
        )
        self.values_ = values
        self.length_ = lengths
        self.dilation_ = dilations
        self.normalize_ = normalize
        self.threshold_ = threshold
        
        if self.classifier is None:
            self.classifier = make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(
                    alphas=np.logspace(-8,8,20),
                    class_weight='balanced', 
                    fit_intercept=True
                )
            )
        self.classifier = self.classifier.fit(out, y)
        return self

    def predict(self, X):
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
        return self.classifier.predict(X_new)

    def _check_params(self, n_timestamps):
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

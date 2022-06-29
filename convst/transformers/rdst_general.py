# -*- coding: utf-8 -*-

from numba import njit, prange
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from convst.utils.checks_utils import check_array_3D, check_array_1D
from convst.utils.shapelets_utils import generate_strides_1D
import pandas as pd
from convst.utils.checks_utils import check_n_jobs
from numba import set_num_threads


@njit(fastmath=True, cache=True, error_model='numpy')
def compute_shapelet_dist_vector(x, values, length, dilation, normalize):
    """
    Compute a shapelet distance vector from an univariate time series 
    and a dilated shapelet.
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
        for j in range(length):
            s += abs(x0[j] - values[j])
        x_conv[i] = s
    return x_conv


@njit(cache=True, parallel=True)
def _init_random_shapelet_params(n_shapelets, shapelet_sizes, n_timestamps, p_norm,
                                 n_features, max_channels):
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
    lengths = np.random.choice(shapelet_sizes, size=n_shapelets).astype(np.int64)
    
    # Channels (features to consider)
    channels = np.zeros((n_shapelets, n_features), dtype=np.bool_)
    n_channels = np.random.choice(np.arange(1,max_channels+1), size=n_shapelets)
    
    # Dilations
    upper_bounds = np.log2(np.floor_divide(n_timestamps - 1, lengths - 1))
    powers = np.empty(n_shapelets)
    for i in prange(n_shapelets):
        powers[i] = np.random.uniform(0, upper_bounds[i])
        
        i_channels = np.random.choice(
            n_features, size=n_channels[i], replace=False
        )
        for j in i_channels:
            channels[i, j] = True
        
    
    dilations = np.floor(np.power(2, powers)).astype(np.int64)

    # Init threshold array
    threshold = np.zeros(n_shapelets, dtype=np.float64)

    # Init values array
    values = np.zeros(
        (n_shapelets, n_features, np.int64(np.max(shapelet_sizes))),
        dtype=np.float64)

    # Is shapelet using z-normalization ?
    normalize = np.random.random(size=n_shapelets)
    normalize = (normalize < p_norm).astype(np.float64)

    return values, lengths, dilations, threshold, normalize, channels


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
    for j in range(l):
        v[j] = X[_idx]
        _sum += X[_idx]
        _sum2 += X[_idx]**2
        _idx += d
    #0 if normalize, seems faster than adding a if statement
    mean = (_sum/l)*normalize
    _std = (_sum2/l) - (mean**2)
    if _std < 0:
        _std = 0
    std = (np.sqrt(_std)+1e-8)*normalize
    # divided by 1 if non normalized
    v = (v - mean)/(std + 1-normalize)
    return v


@njit(cache=True, parallel=True)
def generate_shapelet(X, y, n_shapelets, shapelet_sizes, min_len,
                      seed, p_norm, p_min, p_max, X_len, max_channels):
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
    X_len : array, shape=(n_samples)
        Real number of timestamp of each input time series
        
    Returns
    -------
    set of arrays
        Return the values, lengths, dilations, thresholds and normalization
        indicators as array with first dimension of size (n_shapelets)
    """
    n_samples, n_features, _ = X.shape

    # Fix the random see
    np.random.seed(seed)

    values, lengths, dilations, threshold, normalize, channels = _init_random_shapelet_params(
        n_shapelets, shapelet_sizes, min_len, p_norm, n_features, max_channels
    )

    samples_pool = np.arange(X.shape[0]).astype(np.int64)
    np.random.shuffle(samples_pool)
    # For Values, draw from random uniform (0,n_samples*(n_ts-(l-1)*d))
    # for each l,d combinations. Then take by index the values instead
    # of generating strides.
                
            
    for i in prange(n_shapelets):
        id_sample = samples_pool[i % X.shape[0]]
        index = np.int64(np.random.choice(
            X_len[id_sample] - (lengths[i]-1)*dilations[i]
        ))
        id_test = np.random.choice(np.where(y == y[id_sample])[0])
        
        x_dist = np.zeros(X_len[id_test] - (lengths[i]-1)*dilations[i])
        for j in range(n_features):
            if channels[i,j]:
                values[i, j, :lengths[i]] = _get_subsequence(
                    X[id_sample, j, :X_len[id_sample]], 
                    index, lengths[i], dilations[i], normalize[i]
                )
                x_dist += compute_shapelet_dist_vector(
                    X[id_test, j, :X_len[id_test]],
                    values[i,j], lengths[i], dilations[i], normalize[i]
                )

        threshold[i] = np.random.uniform(
            np.percentile(x_dist, p_min), np.percentile(x_dist, p_max)
        )

    return values, lengths, dilations, threshold, normalize.astype(np.int64), channels


@njit(cache=True, parallel=True, fastmath=True, error_model='numpy')
def apply_all_shapelets(X, values, lengths, dilations, threshold,
                        normalize, channels, X_len):
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
        Normalization indicator of the shapelets
    channels : array, shape=(n_shapelets, n_features)
        Features targeted by each shapelet
    X_len : array, shape=(n_samples)
        Real number of timestamp of each input time series
        
    Returns
    -------
    X_new : array, shape=(n_samples, 4*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.
    """
    n_samples, n_ft, _ = X.shape
    n_shapelets = len(lengths)
    n_features = 4
    
    unique_lengths = np.unique(lengths)
    unique_dilations = np.unique(dilations)

    X_new = np.empty((n_samples, n_features * n_shapelets), dtype=np.float64)
    for index_l in range(unique_lengths.shape[0]):
        l = unique_lengths[index_l]
        for index_d in prange(unique_dilations.shape[0]):
            d = unique_dilations[index_d]
            ix_shapelets = np.where((lengths == l) & (dilations == d))[0]
            if len(ix_shapelets) > 0:
                for i in prange(n_samples):
                    d_shape = X_len[i] - (l-1)*d
                    X_dist = np.zeros((len(ix_shapelets), d_shape))
                    for i_ft in range(n_ft):
                        X_sample = np.zeros((2, d_shape, l))
                        strides = generate_strides_1D(X[i, i_ft, :X_len[i]], l, d)
                        X_sample[0] = strides
                        X_sample[1] = strides
                        for j in range(d_shape):
                            X_sample[1, j] = (X_sample[1, j] - np.mean(X_sample[1, j]))/(
                                np.std(X_sample[1, j])+1e-8
                            )
                        
                        for j in range(len(ix_shapelets)):
                            i_shp = ix_shapelets[j]
                            if channels[i_shp,i_ft]:
                                X_dist[j] += _get_dist_vect(X_sample[normalize[i_shp]],  values[i_shp, i_ft])                                
                    for j in range(len(ix_shapelets)):
                        i_shp = ix_shapelets[j]
                        X_new[i, (n_features * i_shp):(n_features * i_shp + n_features)] = extract_features(
                             X_dist[j], threshold[i_shp]
                        )
    return X_new

@njit(fastmath=True, cache=True, error_model='numpy')
def _get_dist_vect(x, values):
    n_candidates, length = x.shape
    x_dist = np.zeros(n_candidates)
    #For each step of the moving window in the shapelet distance
    for i in range(n_candidates):
        _dist = 0
        #For each value of the shapelet
        for j in prange(length):
            _dist += abs(x[i, j] - values[j])
        x_dist[i] += _dist
    return x_dist

@njit(fastmath=True, cache=True)
def extract_features(x, threshold):
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

    _min = np.min(x)
    _argmin = np.where(x == _min)[0][-1] / x.shape[0]
    _n_occ = np.float64(np.sum(x<threshold))
    _last_occ = -1.0
    if _n_occ > 0.:
        _last_occ = np.float64(np.where(x<threshold)[0][-1]) / x.shape[0]

    return _min, _argmin, _n_occ, _last_occ
    #return np.min(x), np.float64(np.sum(x<threshold))


class GR_DST(BaseEstimator, TransformerMixin):
    """
    Implementation of univariate Random Dilated Shapelet Transform (RDST).
    For details and explanation on the algorithm, users are referred to [1]_:
        
    Parameters
    ----------
    
    random_state : int, optional
        Value of the random state for trees. The default is None.
        
    Attributes
    ----------
    
    dilation : array, shape=(n_shapelets)
        The dilation parameter of each shapelets will be stored in this
        array after calling the fit method.
        
    
    Notes
    -----
    .. [1] Antoine Guillaume et al, "Random Dilated Shapelet Transform: A new approach of time series shapelets" (2022)
    
    
    """
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11], max_channels=None,
                 random_state=None, p_norm=0.8, percentiles=[5, 10], n_jobs=1, min_len=None):
        self.n_shapelets = n_shapelets
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.p_norm = p_norm
        self.max_channels = max_channels
        self.percentiles = percentiles
        self.min_len=min_len
        self.n_jobs = check_n_jobs(n_jobs)
        set_num_threads(n_jobs)
    
    def fit(self, X, y=None):
        
        X, X_len = self._format_uneven_timestamps(X)
        if self.min_len is None:
            self.min_len = X_len.min()
        else:
            self.min_len = min(X_len.min(), self.min_len)
        self.min_len = np.int64(self.min_len)
        X = check_array_3D(X).astype(np.float64)
        n_samples, n_features, n_timestamps = X.shape
        
        if self.shapelet_sizes.dtype == float:
            self.shapelet_sizes = np.floor(n_timestamps*self.shapelet_sizes)
        shapelet_sizes, seed = self._check_params(self.min_len)
        if self.max_channels is None:
            self.max_channels = X.shape[1]
        else:
            self.max_channels = int(self.max_channels)
        # Generate the shapelets
        
        values, lengths, dilations, threshold, normalize, channels = generate_shapelet(
            X, y, self.n_shapelets, shapelet_sizes, self.min_len , seed, self.p_norm,
            self.percentiles[0], self.percentiles[1], X_len, self.max_channels
        )
        self.values_ = values
        self.length_ = lengths
        self.dilation_ = dilations
        self.normalize_ = normalize
        self.threshold_ = threshold
        self.channels_ = channels
        
        return self

    def transform(self, X):
        
        X, X_len = self._format_uneven_timestamps(X)
        X = check_array_3D(X).astype(np.float64)
        check_is_fitted(self, ['values_', 'length_','channels_',
                        'dilation_', 'threshold_', 'normalize_'])
        
        X_new = apply_all_shapelets(
            X, self.values_, self.length_, self.dilation_, self.threshold_,
            self.normalize_, self.channels_, X_len
        )
        return X_new

    def _format_uneven_timestamps(self, X):
        n_ft = np.zeros(len(X),dtype=np.int64)
        lengths = np.zeros(len(X),dtype=np.int64)
        for i in range(len(X)):
            if isinstance(X[i], pd.DataFrame):
                X[i] = X[i].values.T
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
            shapelet_sizes = shapelet_sizes[shapelet_sizes <= n_timestamps]

            raise ValueError("All the values in 'shapelet_sizes' must be lower "
                             "than or equal to 'n_timestamps//2' ({} > {})."
                             .format(shapelet_sizes.max(), n_timestamps//2))
            
        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint64).max, dtype='u8')

        return shapelet_sizes, seed

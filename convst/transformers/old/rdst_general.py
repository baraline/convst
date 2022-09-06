# -*- coding: utf-8 -*-
"""
@author: Antoine Guillaume
"""
import numpy as np
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils import resample

from convst.utils.checks_utils import check_array_3D, check_array_1D,check_n_jobs

from numba import set_num_threads
from numba import njit, prange

@njit(cache=True, error_model='numpy')
def _generate_strides_1D_phase(x, l, d):
    n_ts = x.shape[0]
    x_new = np.zeros((n_ts,l))
    for i in prange(n_ts):
        for j in prange(l):
            x_new[i,j] = x[(i+(j*d))%n_ts]
    return x_new


@njit(cache=True, error_model='numpy')
def _generate_strides_2D_phase(x, l, d):
    n_ft, n_ts = x.shape
    x_new = np.zeros((n_ft, n_ts, l))
    for ft in prange(n_ft):
        for i in prange(n_ts):
            for j in prange(l):
                x_new[ft,i,j] = x[ft,(i+(j*d))%n_ts]
    return x_new


@njit(fastmath=True, cache=True, error_model='numpy')
def compute_shapelet_dist_vector(x, values, length, dilation):
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
    c = _generate_strides_1D_phase(x, length, dilation)
    x_conv = np.empty(c.shape[0])
    for i in range(x_conv.shape[0]):
        s = 0
        for j in prange(length):
            s += abs(c[i,j] - values[j])
        x_conv[i] = s
    return x_conv

@njit(fastmath=True, cache=True, error_model='numpy')
def compute_shapelet_dist_vector_norm(x, values, length, dilation):
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
    c = _generate_strides_1D_phase(x, length, dilation)
    x_conv = np.empty(c.shape[0])
    for i in range(x_conv.shape[0]):
        s = 0
        x0 = (c[i] - c[i].mean())/(c[i].std()+1e-8)
        for j in prange(length):
            s += abs(x0[j] - values[j])
        x_conv[i] = s
    return x_conv

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
    n_ts = X.shape[0]
    
    v = np.empty(l, dtype=np.float64)
    _idx = i_start
    
    for j in prange(l):
        v[j] = X[_idx]
        _idx = (_idx+d)%n_ts
    #0 if normalize, seems faster than adding a if statement
    if normalize:
        v = (v - v.mean())/(v.std()+1e-8)
    return v

@njit(cache=True)
def _init_random_shapelet_params(
        n_shapelets, shapelet_sizes, n_timestamps, p_norm, max_channels
    ):
    """
    Initialize the parameters of the shapelets.    

    Parameters
    ----------
    n_shapelets : int
        Number of shapelet to initialize
    shapelet_sizes : array, shape=()
        Set of possible length for the shapelets
    n_features : int
        The number of features of the dataset
    n_timestamps : int
        Number of timestamps in the input data
    p_norm : float
        A value in the range [0,1] indicating the chance for each
        shapelet to use z-normalized distance
    max_channels : int
        The maximum number of features considered for one shapelet
    
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
    channels : array, shape=(n_shapelet, n_features)
        The features considered by each shapelet
    """
    # Lengths of the shapelets
    lengths = np.random.choice(shapelet_sizes, size=n_shapelets).astype(np.int64)

    # Dilations
    upper_bounds = np.log2(np.floor_divide(n_timestamps - 1, lengths - 1))
    powers = np.empty(n_shapelets)
    for i in prange(n_shapelets):
        powers[i] = np.random.uniform(0, upper_bounds[i])
    dilations = np.floor(np.power(2, powers)).astype(np.int64)

    # Init threshold array
    threshold = np.zeros(n_shapelets, dtype=np.float64)
    
    # channels (i.e. features)
    n_channels = np.random.choice(max_channels, size=n_shapelets)+1

    channel_ids = np.zeros(n_channels.sum(), dtype=np.int64)

    # Init values array
    values = np.zeros(
        np.int64(
            np.dot(lengths.astype(np.float64), n_channels.astype(np.float64))
        ),
        dtype=np.float64,
    )
    
    # Is shapelet using z-normalization ?
    normalize = np.random.random(size=n_shapelets)
    normalize = (normalize < p_norm)

    return values, lengths, dilations, threshold, normalize, n_channels, channel_ids


@njit(cache=True, parallel=True)
def generate_shapelet(
        X, y, n_shapelets, shapelet_sizes, seed, p_norm,
        p_min, p_max, alpha, max_channels, min_len ,X_len
    ):
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
    alpha : float
        Alpha similarity parameter
        
    Returns
    -------
    set of arrays
        Return the values, lengths, dilations, thresholds, normalization and channels
        indicators as array with first dimension of size (n_shapelets)
    """
    n_samples, n_features, _ = X.shape
    # Fix the random see
    np.random.seed(seed)
    values, lengths, dilations, threshold, normalize, n_channels, channel_ids = _init_random_shapelet_params(
        n_shapelets, shapelet_sizes, min_len, p_norm, max_channels
    )
    a1 = np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(n_channels*lengths)))
    a2 = np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(n_channels)))
    
    for i in prange(n_shapelets):
        
        _length = lengths[i]
        _n_channels = n_channels[i]
        _dilation = dilations[i]
        _norm = normalize[i]
        
        _values = np.zeros(_n_channels * _length).astype(
            np.float64
        )
        
        _channel_ids = np.random.choice(
            np.arange(0, n_features), _n_channels, replace=False
        )
        
        _i_sample = np.random.choice(n_samples)
        _n_timestamps_s = X_len[_i_sample]
        _i_timestamp = np.random.choice(_n_timestamps_s)
        
        _i_test = np.random.choice(np.where(y==y[_i_sample])[0])
        _n_timestamps_t = X_len[_i_test]
        _x_dist = np.zeros(_n_timestamps_t)
        
        a3 = 0  # for weights (per channel)
        for _i_c in range(_n_channels):
            b3 = a3 + _length
            _v = _get_subsequence(
                X[_i_sample, _channel_ids[_i_c], :_n_timestamps_s], _i_timestamp,
                _length, _dilation, _norm
            )
            if _norm:
                _x_dist += compute_shapelet_dist_vector_norm(
                    X[_i_test, _channel_ids[_i_c], :_n_timestamps_t], _v, _length, _dilation    
                )
                
            else:
                _x_dist += compute_shapelet_dist_vector(
                    X[_i_test, _channel_ids[_i_c], :_n_timestamps_t], _v, _length, _dilation    
                )

            _values[a3:b3] = _v
            a3 = b3
        
        values[a1[i]:a1[i+1]] = _values
        channel_ids[a2[i]:a2[i+1]] = _channel_ids
        threshold[i] = np.random.uniform(
            np.percentile(_x_dist,p_min),
            np.percentile(_x_dist,p_max)
        )    
        
    
    return (
        values,
        lengths,
        dilations,
        threshold,
        normalize.astype(np.int64),
        n_channels,
        channel_ids
    )


@njit(cache=True)
def _combinations_1d(x,y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    mesh = np.empty(shape=(x_size * y_size, 2), dtype=x.dtype)
    for i in prange(x.size):
        for j in prange(y.size):
            mesh[i*x_size + j, 0] = x[i]
            mesh[i*x_size + j, 1] = y[j]
    return mesh

@njit(cache=True, parallel=True, fastmath=True, error_model='numpy')
def apply_all_shapelets(X, X_len, shapelets):
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
    channels : array, shape=(n_shapelet, n_features)
        The features considered by each shapelet

    Returns
    -------
    X_new : array, shape=(n_samples, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.

    """
    (values, lengths, dilations, threshold, normalize, n_channels, channel_ids) = shapelets
    n_shapelets = len(lengths)
    n_samples, n_ft, _ = X.shape
    n_features = 3

    unique_lengths = np.unique(lengths)
    unique_dilations = np.unique(dilations)
    
    #(u_l * u_d , 2)
    params_shp = _combinations_1d(unique_lengths, unique_dilations)
    #(u_l * u_d) + 1
    n_shp_params = np.zeros(params_shp.shape[0]+1, dtype=np.int64)
    #(n_shapelets)
    idx_shp = np.zeros(n_shapelets, dtype=np.int64)
    
    a1 = np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(n_channels*lengths)))
    a2 = np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(n_channels)))
    
    a = 0
    for i in range(params_shp.shape[0]):
        _length = params_shp[i, 0]
        _dilation = params_shp[i, 1]
        
        ix_shapelets = np.where((lengths == _length) & (dilations == _dilation))[0]
        b = a + ix_shapelets.shape[0]
        
        idx_shp[a:b] = ix_shapelets
        n_shp_params[i+1] = ix_shapelets.shape[0]
        
        a = b
    
    n_shp_params = np.cumsum(n_shp_params)
    X_new = np.empty((n_samples, n_features * n_shapelets), dtype=np.float64)
    for i_sample in prange(n_samples):
        _n_timestamps = X_len[i_sample]
        
        #n_shp_params is a cumsum starting at 0
        for i_shp_param in prange(n_shp_params.shape[0]-1):
            _length = params_shp[i_shp_param, 0]
            _dilation = params_shp[i_shp_param, 1]
            
            strides = _generate_strides_2D_phase(
                X[i_sample, :, :_n_timestamps], _length, _dilation
            )
            # Indexes of shapelets corresponding to the params of i_shp_param
            _idx_shp = idx_shp[n_shp_params[i_shp_param]:n_shp_params[i_shp_param+1]]
            
            _idx_no_norm = _idx_shp[np.where(normalize[_idx_shp] == 0)[0]]
            for i_idx in range(_idx_no_norm.shape[0]):               
                i_shp = _idx_no_norm[i_idx]
                _channels = channel_ids[a2[i_shp]:a2[i_shp+1]]
                _values = values[a1[i_shp]:a1[i_shp+1]].reshape(n_channels[i_shp], _length)
                
                X_new[i_sample, (n_features * i_shp):(n_features * i_shp + n_features)] = apply_one_shapelet_one_sample(
                    strides[_channels], _values, threshold[i_shp],
                )
            
            _idx_norm = _idx_shp[np.where(normalize[_idx_shp] == 1)[0]]
            if _idx_norm.shape[0] > 0:
                #n_features
                for i_stride in range(strides.shape[0]):
                    #n_timestamps
                    for j_stride in range(strides.shape[1]):
                        _str = strides[i_stride, j_stride]
                        strides[i_stride, j_stride] = _str - np.mean(_str)/(np.std(_str)+1e-8)
                        
                for i_idx in range(_idx_norm.shape[0]):               
                    i_shp = _idx_norm[i_idx]
                    _channels = channel_ids[a2[i_shp]:a2[i_shp+1]]
                    _values = values[a1[i_shp]:a1[i_shp+1]].reshape(n_channels[i_shp], _length)
                    
                    X_new[i_sample, (n_features * i_shp):(n_features * i_shp + n_features)] = apply_one_shapelet_one_sample(
                        strides[_channels], _values, threshold[i_shp],
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
    x : array, shape=()
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
    n_ft, n_candidates, length = x.shape

    _n_match = 0
    _min = 1e+100
    _argmin = -1.0

    #For each step of the moving window in the shapelet distance
    for i in range(n_candidates):
        _dist = 0
        for ft in prange(n_ft):
            #For each value of the shapelet
            for j in prange(length):
                _dist += abs(x[ft, i, j] - values[ft, j])

        if _dist < _min:
            _min = _dist
            _argmin = i
            
        if _dist <= threshold:
            _n_match += 1

    return _min, np.float64(_argmin), np.float64(_n_match)


class GR_DST(BaseEstimator, TransformerMixin):
    #TODO : add reference to thesis for new additions
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
    alpha : float, optional
        The alpha-similarity parameter between 0 and 1. During sampling, if at 
        least alpha percent of its indexes are already used by a previously
        sampled shapelet with similar parameter, it will be ignored. If the whole
        dataset is covered, this can cause the final number of shapelet to be
        less than n_shapelets.
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
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11], n_jobs=1, n_samples=None,
                 p_norm=0.8, percentiles=[5, 10], random_state=None, alpha=0.5,
                 max_channels=None, min_len=None):
        self.n_shapelets = n_shapelets
        self.min_len = min_len
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.p_norm = p_norm
        self.max_channels = max_channels
        self.n_samples=n_samples
        self.alpha=alpha
        self.percentiles = percentiles
        self.n_jobs = check_n_jobs(n_jobs)
        set_num_threads(n_jobs)
        
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
        X, X_len = self._format_uneven_timestamps(X)
        if self.min_len is None:
            self.min_len = X_len.min()
        X = check_array_3D(X, is_univariate=False).astype(np.float64)
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
        n_samples, n_features, n_timestamps = X.shape
        
        if self.shapelet_sizes.dtype == float:
            self.shapelet_sizes = np.floor(self.min_len*self.shapelet_sizes)
            
        if self.max_channels is None:
            self.max_channels = n_features
        elif isinstance(self.max_channels, float):
            self.max_channels = max(1,int(self.max_channels*n_features))
        elif isinstance(self.max_channels, int):
            pass
        shapelet_sizes, seed = self._check_params(self.min_len)
        # Generate the shapelets

        self.shapelets_ = generate_shapelet(
            X, y, self.n_shapelets, shapelet_sizes, seed, self.p_norm,
            self.percentiles[0], self.percentiles[1], self.alpha, self.max_channels,
            self.min_len, X_len
        )
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
        X, X_len = self._format_uneven_timestamps(X)
        X = check_array_3D(X, is_univariate=False).astype(np.float64)
        check_is_fitted(self, ['shapelets_'])
        X_new = apply_all_shapelets(
            X, X_len, self.shapelets_ 
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
            if n_timestamps < 5:
                raise ValueError('Input data goint {} timestamps, at least 5 are requiered. Input format should be (n_samples, n_features, n_timestamps)'.format(n_timestamps))
            else:
                warnings.warn("All the values in 'shapelet_sizes' must be lower than or equal to 'n_timestamps' (got {} > {}). Changed shapelet size to {}".format(shapelet_sizes.max(), n_timestamps, n_timestamps//2))
                shapelet_sizes = np.array([n_timestamps//2])


        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

        return shapelet_sizes, seed
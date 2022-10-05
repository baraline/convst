# -*- coding: utf-8 -*-

from numpy.lib.stride_tricks import as_strided
from numba import njit, prange
from numpy import float_, sqrt, zeros

###############################################################################
#                                                                             #
#                         DISTANCE FUNCTIONS                                  #
#                                                                             #
###############################################################################

@njit(
  fastmath=True, cache=True
)
def euclidean(x, y):
    s = 0
    for i in prange(x.shape[0]):
        s += (x[i]-y[i])**2
    return sqrt(s)

@njit(
  fastmath=True, cache=True
)
def squared_euclidean(x, y):
    s = 0
    for i in prange(x.shape[0]):
        s += (x[i]-y[i])**2
    return s

@njit(
  fastmath=True, cache=True
)
def manhattan(x, y):
    s = 0
    for i in prange(x.shape[0]):
        s += abs(x[i]-y[i])
    return s

###############################################################################
#                                                                             #
#                    SUBSEQUENCE EXTRACTION FUNCTIONS                         #
#                                                                             #
###############################################################################

@njit(
  cache=True
)
def generate_strides_1D(X, window_size, dilation, use_phase):
    if use_phase:
        return _generate_strides_1D_phase(X, window_size, dilation) 
    else:
        return _generate_strides_1D(X, window_size, dilation)

@njit(
  cache=True
)

def generate_strides_2D(X, window_size, dilation, use_phase):
    if use_phase:
        return _generate_strides_2D_phase(X, window_size, dilation) 
    else:
        return _generate_strides_2D(X, window_size, dilation)


@njit(
  cache=True
)
def _generate_strides_1D(X, window_size, dilation):
    """
    Generate strides from the input univariate time series with specified 
    length and dilation parameters.

    Parameters
    ----------
    X : array, shape = (n_timestamps)
        An univariate time series, in a 1 dimensional view.
    window_size : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_timestamps-(window_size-1)*dilation, window_size)
        All possible subsequences of length stride_len for each time series.

    """
    n_timestamps = X.shape[0] - (window_size - 1) * dilation
    X_new = zeros((n_timestamps, window_size))
    for i in prange(n_timestamps):
        for j in prange(window_size):
            X_new[i,j] = X[i+(j*dilation)]
    return X_new

@njit(
  cache=True
)

def _generate_strides_2D(X, window_size, dilation):
    """
    Generate strides from an ensemble of univariate time series with specified 
    length and dilation parameters.

    Parameters
    ----------
    X : array, shape = (n_samples, n_timestamps)
        An ensemble of univariate time series, in a 2 dimensional view. It can
        also be a multivariate time series as (n_features, n_timestamps).
    window_size : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_samples, n_timestamps-(window_size-1)*dilation, window_size)
        All possible subsequences of length stride_len for each time series.
    """
    n_features, n_timestamps = X.shape
    n_timestamps = n_timestamps - (window_size - 1) * dilation
    X_new = zeros((n_features, n_timestamps, window_size))
    for ft in prange(n_features):
        for i in prange(n_timestamps):
            for j in prange(window_size):
                X_new[ft,i,j] = X[ft,i+(j*dilation)]
    return X_new


@njit(
  cache=True
)
def _generate_strides_1D_phase(X, window_size, dilation):
    """
    Generate strides from the input univariate time series with specified 
    length and dilation parameters. It uses phase invariance.

    Parameters
    ----------
    X : array, shape = (n_timestamps)
        An univariate time series, in a 1 dimensional view.
    window_size : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_timestamps, window_size)
        All possible subsequences of length stride_len for each time series.
    """
    n_timestamps = X.shape[0]
    X_new = zeros((n_timestamps,window_size))
    for i in prange(n_timestamps):
        for j in prange(window_size):
            X_new[i,j] = X[(i+(j*dilation))%n_timestamps]
    return X_new


@njit(cache=True)
def _generate_strides_2D_phase(X, window_size, dilation):
    """
    Generate strides from an ensemble of univariate time series with specified 
    length and dilation parameters. It uses phase invariance.

    Parameters
    ----------
    X : array, shape = (n_samples, n_timestamps)
        An ensemble of univariate time series, in a 2 dimensional view. It can
        also be a multivariate time series as (n_features, n_timestamps).
    window_size : int
        Length of the strides to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_samples, n_timestamps, window_size)
        All possible subsequences of length stride_len for each time series.
    """
    n_features, n_timestamps = X.shape
    X_new = zeros((n_features, n_timestamps, window_size))
    for ft in prange(n_features):
        for i in prange(n_timestamps):
            for j in prange(window_size):
                X_new[ft,i,j] = X[ft,(i+(j*dilation))%n_timestamps]
    return X_new


@njit(cache=True)
def get_subsequence(X, index, length, d, normalize, use_phase):
    if use_phase:
        return _get_subsequence_phase(
            X, index, length, d, normalize
        )
    else:
        return _get_subsequence(
            X, index, length, d, normalize
        )


@njit(cache=True, fastmath=True)
def _get_subsequence(X, i_start, length, d, normalize):
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
    normalize : bool
        Boolean indicating wheter or not the distance will be z-normalized

    Returns
    -------
    v : array, shape=(length)
        The resulting subsequence.

    """
    v = zeros(length)
    _idx = i_start
    
    for j in prange(length):
        v[j] = X[_idx]
        _idx = _idx + d
    
    if normalize:
        v = (v - v.mean())/(v.std()+1e-8)
    return v

@njit(cache=True, fastmath=True)
def _get_subsequence_phase(X, i_start, length, d, normalize):
    """
    Given a set of length and dilation, fetch a subsequence from an input 
    time series given a starting index. It uses phase invariance.

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
    normalize : bool
        Boolean indicating wheter or not the distance will be z-normalized


    Returns
    -------
    v : array, shape=(length)
        The resulting subsequence.

    """
    n_ts = X.shape[0]
    v = zeros(length)
    _idx = i_start
    
    for j in prange(length):
        v[j] = X[_idx]
        _idx = (_idx+d)%n_ts
    
    if normalize:
        v = (v - v.mean())/(v.std()+1e-8)
    return v

###############################################################################
#                                                                             #
#                       DISTANCE VECTOR COMPUTATION                           #
#                                                                             #
###############################################################################

@njit(cache=True)
def compute_shapelet_dist_vector(
    x, values, length, dilation, dist_func, normalize, use_phase
):
    if normalize and use_phase:
        return _compute_shapelet_dist_vector_norm_phase(
            x, values, length, dilation, dist_func
        )
    elif normalize and not use_phase:
        return _compute_shapelet_dist_vector_norm(
            x, values, length, dilation, dist_func
        )
    elif not normalize and use_phase:
        return _compute_shapelet_dist_vector_phase(
            x, values, length, dilation, dist_func
        )
    elif not normalize and not use_phase:
        return _compute_shapelet_dist_vector(
            x, values, length, dilation, dist_func
        )
    else:
        raise ValueError('Wrong parameter for normalize or phase')

@njit(fastmath=True, cache=True)
def _compute_shapelet_dist_vector(x, values, length, dilation, dist_func):
    """
    Compute a shapelet distance vector from an univariate time series 
    and a dilated shapelet. Shapelet should be already normalized if normalizing
    the distance

    Parameters
    ----------
    x : array, shape=(n_timestamps)
        An input time series
    values : array, shape=(length)
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
    c = _generate_strides_1D(x, length, dilation)
    x_conv = zeros(c.shape[0])
    for i in prange(x_conv.shape[0]):
        x_conv[i] = dist_func(c[i], values)
    return x_conv

@njit(fastmath=True, cache=True)
def _compute_shapelet_dist_vector_norm(x, values, length, dilation, dist_func):
    """
    Compute a shapelet distance vector from an univariate time series 
    and a dilated shapelet. Shapelet should be already normalized if normalizing
    the distance.

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
    c = _generate_strides_1D(x, length, dilation)
    x_conv = zeros(c.shape[0])
    for i in prange(x_conv.shape[0]):
        x0 = (c[i] - c[i].mean())/(c[i].std()+1e-8)
        x_conv[i] = dist_func(x0, values)
    return x_conv

@njit(fastmath=True, cache=True)
def _compute_shapelet_dist_vector_phase(x, values, length, dilation, dist_func):
    """
    Compute a shapelet distance vector from an univariate time series 
    and a dilated shapelet. Shapelet should be already normalized if normalizing
    the distance. It uses phase invariance.

    Parameters
    ----------
    x : array, shape=(n_timestamps)
        An input time series
    values : array, shape=(length)
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
    x_conv : array, shape=(n_timestamps)
        The resulting distance vector

    """
    c = _generate_strides_1D_phase(x, length, dilation)
    x_conv = zeros(c.shape[0])
    for i in prange(x_conv.shape[0]):
        x_conv[i] = dist_func(c[i], values)
    return x_conv

@njit(fastmath=True, cache=True)
def _compute_shapelet_dist_vector_norm_phase(x, values, length, dilation, dist_func):
    """
    Compute a shapelet distance vector from an univariate time series 
    and a dilated shapelet. Shapelet should be already normalized if normalizing
    the distance. It uses phase invariance.

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
    x_conv : array, shape=(n_timestamps)
        The resulting distance vector

    """
    c = _generate_strides_1D_phase(x, length, dilation)
    x_conv = zeros(c.shape[0])
    for i in prange(x_conv.shape[0]):
        x0 = (c[i] - c[i].mean())/(c[i].std()+1e-8)
        x_conv[i] = dist_func(x0, values)
    return x_conv


@njit(fastmath=True, cache=True)
def apply_one_shapelet_one_sample_univariate(x, values, threshold, dist_func):
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
        _dist = dist_func(x[i], values)

        if _dist < _min:
            _min = _dist
            _argmin = i

        if _dist <= threshold:
            _n_match += 1
            
    return _min, float_(_argmin), float_(_n_match)

@njit(fastmath=True, cache=True)
def apply_one_shapelet_one_sample_multivariate(x, values, threshold, dist_func):
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
    n_ft, n_candidates, length = x.shape

    _n_match = 0
    _min = 1e+100
    _argmin = 0
    
    #For each step of the moving window in the shapelet distance
    for i in range(n_candidates):
        _dist = 0
        for ft in prange(n_ft):
            _dist += dist_func(x[ft, i], values[ft])
    
        if _dist < _min:
            _min = _dist
            _argmin = i
            
        if _dist <= threshold:
            _n_match += 1
    
    return _min, float_(_argmin), float_(_n_match)


@njit(cache=True)
def _combinations_1d(x,y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    mesh = zeros((x_size * y_size, 2), dtype=x.dtype)
    for i in prange(x.size):
        for j in prange(y.size):
            mesh[i*x_size + j, 0] = x[i]
            mesh[i*x_size + j, 1] = y[j]
    return mesh
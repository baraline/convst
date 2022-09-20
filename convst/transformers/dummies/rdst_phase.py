# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:17:07 2021

@author: a694772
"""
import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils import resample

from convst.utils.checks_utils import check_array_3D, check_array_1D


from numba import set_num_threads
from numba import njit, prange
from convst.utils.checks_utils import check_n_jobs

@njit(cache=True, error_model='numpy')
def _generate_strides_1D_phase(x, l, d):
    n_ts = x.shape[0]
    x_new = np.zeros((n_ts,l))
    for i in prange(n_ts):
        for j in prange(l):
            x_new[i,j] = x[(i+(j*d))%n_ts]
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


@njit(cache=True, parallel=True)
def generate_shapelet(X, y, n_shapelets, shapelet_sizes, seed, p_norm, p_min, p_max, alpha):
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
    unique_dil = np.unique(dilations)
    mask_sampling = np.ones((2,unique_dil.shape[0],n_samples,n_timestamps),dtype=np.bool_)
    # For Values, draw from random uniform (0,n_samples*(n_ts-(l-1)*d))
    # for each l,d combinations. Then take by index the values instead
    # of generating strides.
    for i_d in prange(unique_dil.shape[0]):
        for i in np.where(dilations==unique_dil[i_d])[0]:
            d = dilations[i]
            l = lengths[i]
            norm = np.int64(normalize[i])
            mask_dil = mask_sampling[norm,i_d]
            i_mask = np.where(mask_dil)
            if i_mask[0].shape[0] > 0:
                id_sample = np.random.choice(i_mask[0])
                index = np.random.choice(i_mask[1][i_mask[0]==id_sample])
                for j in range(np.int64(np.floor(l*alpha))):
                    mask_sampling[norm, i_d, id_sample, (index-(j*d))%n_timestamps] = False
                    mask_sampling[norm, i_d, id_sample, (index+(j*d))%n_timestamps] = False
                
                v = _get_subsequence(
                    X[id_sample, 0], index, l, d, normalize[i]
                )
                values[i, :l] = v
                loc_others = np.where(y == y[id_sample])[0]
                if loc_others.shape[0] > 1:
                    loc_others = loc_others[loc_others != id_sample]
                    id_test = np.random.choice(loc_others)
                else:
                    id_test = id_sample
                if normalize[i]:
                    x_dist = compute_shapelet_dist_vector_norm(
                        X[id_test, 0], values[i], l, d
                    )
                else:
                    x_dist = compute_shapelet_dist_vector(
                        X[id_test, 0], values[i], l, d
                    )
                threshold[i] = np.random.uniform(
                    np.percentile(x_dist, p_min), np.percentile(x_dist, p_max)
                )
                
    mask_values = np.ones(n_shapelets, dtype=np.bool_)
    for i in prange(n_shapelets):
        if np.all(values[i] == 0):
            mask_values[i] = False
    return values[mask_values], lengths[mask_values], dilations[mask_values], threshold[mask_values], normalize.astype(np.int64)[mask_values]




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
            d_shape = n_timestamps

            if len(ix_shapelets) > 0:
                for i in prange(n_samples):
                    strides = _generate_strides_1D_phase(X[i, 0], l, d)
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

        if _dist < threshold:
            _n_match += 1

    return _min, np.float64(_argmin), np.float64(_n_match)


class R_DST_PH(BaseEstimator, TransformerMixin):
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
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11], n_jobs=1, n_samples=None,
                 p_norm=0.8, percentiles=[5, 10], random_state=None, alpha=0.5):
        self.n_shapelets = n_shapelets
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.p_norm = p_norm
        self.n_samples=n_samples
        self.alpha=alpha
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
        if self.n_samples is not None:
            id_X = resample(np.arange(X.shape[0]), replace=False, n_samples=int(X.shape[0]*self.n_samples), stratify=y)
            X = X[id_X]
            y = y[id_X]
        n_samples, n_features, n_timestamps = X.shape
        if self.shapelet_sizes.dtype == float:
            self.shapelet_sizes = np.floor(n_timestamps*self.shapelet_sizes)
        
        shapelet_sizes, seed = self._check_params(n_timestamps)
        # Generate the shapelets

        values, lengths, dilations, threshold, normalize = generate_shapelet(
            X, y, self.n_shapelets, shapelet_sizes, seed, self.p_norm,
            self.percentiles[0], self.percentiles[1], self.alpha
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
            if n_timestamps < 5:
                raise ValueError('Input data goint {} timestamps, at least 5 are requiered. Input format should be (n_samples, n_features, n_timestamps)'.format(n_timestamps))
            else:
                warnings.warn("All the values in 'shapelet_sizes' must be lower than or equal to 'n_timestamps' (got {} > {}). Changed shapelet size to {}".format(shapelet_sizes.max(), n_timestamps, n_timestamps//2))
                shapelet_sizes = np.array([n_timestamps//2])


        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

        return shapelet_sizes, seed

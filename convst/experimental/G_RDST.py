# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:17:07 2021

@author: a694772
"""
from numba import njit, prange, guvectorize, vectorize
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from convst.utils.checks_utils import check_array_3D, check_array_1D
from convst.utils.shapelets_utils import generate_strides_1D
from numba import set_num_threads
import pandas as pd
from matplotlib import pyplot as plt

"""
from convst.utils.dataset_utils import load_sktime_dataset_split
X, Xt, y, yt, _ = load_sktime_dataset_split("StandWalkJump", normalize=True)
X = X.astype(np.float64)
Xt = Xt.astype(np.float64)
a = FR_DST(n_shapelets=2).fit(X, y)
b = a.transform(X)
bt = a.transform(Xt)
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
r = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-6,6,20))
).fit(b, y)
p = r.predict(bt)
print(accuracy_score(yt, p))
"""



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
def generate_shapelet(X, y, n_shapelets, shapelet_sizes,
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
        n_shapelets, shapelet_sizes, X_len.min(), p_norm, n_features, max_channels
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
    X_new : array, shape=(n_samples, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.

    """
    n_samples, n_ft, _ = X.shape
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
        x_dist += _dist
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

    return np.min(x), np.argmin(x)/x.shape[0], np.sum(x<threshold)/x.shape[0]
    #return np.min(x), np.float64(np.sum(x<threshold))


class FR_DST(BaseEstimator, TransformerMixin):
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
                 random_state=None, p_norm=0.8, percentiles=[5, 10]):
        self.n_shapelets = n_shapelets
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.p_norm = p_norm
        self.max_channels = max_channels
        self.percentiles = percentiles

    def fit(self, X, y=None):
        X, X_len = self._format_uneven_timestamps(X)
        X = check_array_3D(X, is_univariate=False).astype(np.float64)
        n_samples, n_features, n_timestamps = X.shape
        if self.shapelet_sizes.dtype == float:
            self.shapelet_sizes = np.floor(n_timestamps*self.shapelet_sizes)
        shapelet_sizes, seed = self._check_params(n_timestamps)
        if self.max_channels is None:
            self.max_channels = X.shape[1]
        else:
            self.max_channels = int(self.max_channels)
        # Generate the shapelets
        
        values, lengths, dilations, threshold, normalize, channels = generate_shapelet(
            X, y, self.n_shapelets, shapelet_sizes, seed, self.p_norm,
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
        X = check_array_3D(X, is_univariate=False).astype(np.float64)
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
            X_new = np.zeros((X.shape[0], n_ft[0], lengths.max()))
            for i in range(X.shape[0]):
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
        if not np.all(shapelet_sizes <= n_timestamps//2):
            shapelet_sizes = shapelet_sizes[shapelet_sizes <= n_timestamps//2]
            if shapelet_sizes.shape[0] == 0:
                shapelet_sizes = np.array([3])
            """
            raise ValueError("All the values in 'shapelet_sizes' must be lower "
                             "than or equal to 'n_timestamps//2' ({} > {})."
                             .format(shapelet_sizes.max(), n_timestamps//2))
            """
        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

        return shapelet_sizes, seed

    def _get_shp_params(self, id_shp):
        #values, length, dilation, padding, range
        return (self.values_[id_shp], self.length_[id_shp],
                self.dilation_[id_shp], self.threshold_[id_shp],
                self.normalize_[id_shp])

    def visualise_one_shapelet(self, id_shp, X, y, target_class,figs=(15, 10)):
        # For visualisation, if argmin is important, draw a bar on x axis
        # If min, highligh on series (red)
        # If #match, hihgligh all parts which match on series (blue)
        sns.set()
        sns.set_context('talk')
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=figs)
        values, length, dilation, r, norm = self._get_shp_params(id_shp)
        values = values[:length]
        X_new = np.zeros((X.shape[0], 3))
        yc = (y == target_class).astype(int)
        for i in range(X.shape[0]):
            x_dist = compute_shapelet_dist_vector(
                X[i, 0], values, length, dilation, norm)
            X_new[i, 0] = np.min(x_dist)
            X_new[i, 1] = np.argmin(x_dist)
            X_new[i, 2] = np.mean(x_dist < r)

        sns.boxplot(x=yc, y=X_new[:, 0], ax=ax[0, 0])
        sns.boxplot(x=yc, y=X_new[:, 1], ax=ax[0, 1])
        sns.boxplot(x=yc, y=X_new[:, 2], ax=ax[0, 2])

        i0 = np.random.choice(np.where(yc == 0)[0])
        i1 = np.random.choice(np.where(yc == 1)[0])
        ax[1, 1].scatter(np.arange(length)*dilation, values)
        ax[1, 1].plot(np.arange(length)*dilation, values, linestyle='--')
        ax[1, 2].plot(compute_shapelet_dist_vector(X[i1, 0], values, length, dilation, norm),
                      c='C1', alpha=0.75, label='distance vector of sample of class {}'.format(target_class))
        ax[1, 2].plot(compute_shapelet_dist_vector(X[i0, 0], values, length, dilation, norm),
                      c='C0', alpha=0.75, label='distance vector of sample of class {}'.format(y[i0]))
        ax[0, 0].set_xticks([0, 1])
        ax[0, 0].set_xticklabels(
            ['other classes', 'class {}'.format(target_class)])
        ax[0, 1].set_xticks([0, 1])
        ax[0, 1].set_xticklabels(
            ['other classes', 'class {}'.format(target_class)])
        ax[0, 2].set_xticks([0, 1])
        ax[0, 2].set_xticklabels(
            ['other classes', 'class {}'.format(target_class)])
        ax[1, 0].set_xlabel('timestamps')
        ax[1, 1].set_xlabel('timestamps')
        ax[1, 2].set_xlabel('timestamps')
        ix_start = X_new[i0, 1]
        ix = np.arange(ix_start, ix_start+length)
        if norm == 1:
            ix = np.zeros(length)
            for i in range(length):
                ix[i] = ix_start + (i*dilation)
            ix = ix.astype(int)
            v = values[:length] * X[i0, 0, ix].std() + X[i0, 0, ix].mean()
        else:
            v = values[:length]
        ax[1, 0].scatter(ix, v, c='C0', alpha=0.75)

        ix_start = X_new[i1, 1]
        ix = np.arange(ix_start, ix_start+length)
        if norm == 1:
            ix = np.zeros(length)
            for i in range(length):
                ix[i] = ix_start + (i*dilation)
            ix = ix.astype(int)
            v = values[:length] * X[i1, 0, ix].std() + X[i1, 0, ix].mean()
        else:
            v = values[:length]
        ax[1, 0].scatter(ix, v, c='C1', alpha=0.75)

        ax[1, 2].axhline(r, c='C2', linestyle='--')
        """
        ax[i,1].set_xticks(np.arange(length), rotation=70)
        ax[i,1].set_xticklabels(np.arange(length)*dilation, rotation=70)
        ax[i,2].plot(x_conv, color = 'C'+str(i))
        ax[i,2].axhline(r)
        """
        ax[1, 0].plot(X[i1, 0], c='C1', alpha=0.75, 
                      label='sample of class {}'.format(target_class))
        ax[1, 0].plot(X[i0, 0], c='C0', alpha=0.75,
                      label='sample of class {}'.format(y[i0]))
        
        ax[0, 0].set_title("Boxplot of min")
        ax[1, 0].set_title("Location of the minimum")
        ax[0, 1].set_title("Boxplot of argmin")
        ax[1, 1].set_title("Shapelet n°{} (d={})".format(id_shp, dilation))
        ax[0, 2].set_title("Boxplot of shapelet occurences")
        ax[1, 2].set_title("Distance vector and lambda threshold")
        #ax[2,1].set_title("0 : {}; 1 : {}".format(str(X_new[i0,2])[0:5],str(X_new[i1,2])[0:5]))
        ax[1, 0].legend()
        ax[1, 1].legend()
        #fig.suptitle("Shapelet l={}, d={}, n={}".format(length,dilation,norm))
        plt.show()

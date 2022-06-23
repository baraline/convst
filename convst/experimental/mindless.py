# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state

from convst.utils.checks_utils import check_array_3D, check_array_1D
from convst.utils.shapelets_utils import generate_strides_1D

from numba import set_num_threads
from numba import njit, prange
from numba.core.config import NUMBA_DEFAULT_NUM_THREADS

from matplotlib import pyplot as plt

@njit(cache=True, fastmath=True)
def _dtw_distance(series1, series2):
    """
    Returns the DTW similarity distance between two 1-D
    timeseries numpy arrays.
    Args:
        series1, series2 : array of shape [n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared.
    Returns:
        DTW distance between A and B
    """
    l1 = series1.shape[0]
    l2 = series2.shape[0]
    E = np.empty((l1, l2))

    # Fill First Cell
    v = series1[0] - series2[0]
    E[0][0] = v * v
    
    # Fill First Column
    for i in prange(1, l1):
        v = series1[i] - series2[0]
        E[i][0] = E[i - 1][0] + v * v

    # Fill First Row
    for i in prange(1, l2):
        v = series1[0] - series2[i]
        E[0][i] = E[0][i - 1] + v * v

    for i in prange(1, l1):
        for j in prange(1, l2):
            v = series1[i] - series2[j]
            v = v * v
    
            v1 = E[i - 1][j]
            v2 = E[i - 1][j - 1]
            v3 = E[i][j - 1]
            
            if v1 <= v2 and v1 <= v3:
                E[i][j] = v1 + v
            elif v2 <= v1 and v2 <= v3:
                E[i][j] = v2 + v
            else:
                E[i][j] = v3 + v
    
    return np.sqrt(E[-1][-1])



@njit(cache=True, parallel=True)
def _init_random_shapelet_params(
        n_shapelets, shapelet_sizes, n_timestamps, invariance_proba
    ):
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
    invariances : array, shape=(n_shapelet,4)
        Which invariance property does the shapelet have ? 
        0: Scale invariance
        1: Phase invariance
        2: Warping invariance
        3: Complexity invariance
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

    # What invariance does the shapelet have ?
    invariances = np.zeros((n_shapelets,4), dtype=np.bool_)
    for i in range(4):
        proba = np.random.random(size=n_shapelets)
        invariances[:,i] = (proba < invariance_proba[i])

    return values, lengths, dilations, threshold, invariances

@njit(cache=True, fastmath=True)
def _diff_sum(X):
    s = 0
    for i in prange(1,X.shape[0]):
        s += abs(X[i] - X[i-1])
    return s

@njit(cache=True)
def sample_shapelet_phase_invariant(X, l, d):
    n_timestamps = X.shape[0]
    v = np.zeros(l)
    start_index = np.int64(np.random.choice(n_timestamps))
    for i in prange(l):
        v[i] += X[(start_index+(i*d))%n_timestamps]
    return v

@njit(cache=True)
def sample_shapelet(X, l, d):
    v = np.zeros(l)
    start_index = np.int64(np.random.choice(X.shape[0] - ((l-1)*d)))
    for i in prange(l):
        v[i] += X[start_index+(i*d)]
    return v
     
@njit(cache=True)
def generate_subs_1D_phase(X, l, d):
    n_timestamps = X.shape[0]
    subs = np.zeros((n_timestamps, l))
    for i in prange(n_timestamps):
        for j in prange(l):
            subs[i, j] = X[(i+(j*d))%n_timestamps]
    return subs

@njit(cache=True, fastmath=True)
def dist_dtw(X, S, CF):
    dist = np.zeros(X.shape[0])
    for i in prange(X.shape[0]):
        dist[i] = _dtw_distance(X[i],S)*CF[i]
    return dist

@njit(cache=True, fastmath=True)
def dist_euc(X, S, CF):
    l = X.shape[1]
    dist = np.zeros(X.shape[0])
    for i in prange(X.shape[0]):
        _s = 0
        for j in prange(l):
            _s += abs(X[i,j] - S[j])
        dist[i] = _s*CF[i]
    return dist

@njit(cache=True, fastmath=True)
def dist_vect(X, S, l, d, inv):
    
    if inv[1]:
        X_subs = generate_subs_1D_phase(X, l ,d)
    else:
        X_subs = generate_strides_1D(X, l, d)
    if inv[0]:
        for i in prange(X_subs.shape[0]):
            X_subs[i] = (X_subs[i] - X_subs[i].mean()) / (X_subs[i].std() + 1e-8)
    CF = np.ones(X_subs.shape[0])
    S_diff = _diff_sum(S)
    if inv[3]:
        for i in prange(X_subs.shape[0]):
            x_diff = _diff_sum(X_subs[i])
            CF[i] = max(x_diff, S_diff)/min(x_diff, S_diff)
    if inv[2]:
        return dist_dtw(X_subs, S, CF)
    else:
        return dist_euc(X_subs, S, CF)

        
@njit(cache=True, parallel=False, fastmath=True)
def generate_shapelet(
        X, y, n_shapelets, shapelet_sizes, seed, invariance_proba, p_min, p_max
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

    Returns
    -------
    set of arrays
        Return the values, lengths, dilations, thresholds and normalization
        indicators as array with first dimension of size (n_shapelets)
    """
    n_samples, n_features, n_timestamps = X.shape

    # Fix the random see
    np.random.seed(seed)

    values, lengths, dilations, threshold, invariances = _init_random_shapelet_params(
        n_shapelets, shapelet_sizes, n_timestamps, invariance_proba
    )

    samples_pool = np.arange(X.shape[0]).astype(np.int64)
    np.random.shuffle(samples_pool)
    # For Values, draw from random uniform (0,n_samples*(n_ts-(l-1)*d))
    # for each l,d combinations. Then take by index the values instead
    # of generating strides.
    
    for i in prange(n_shapelets):
        
        id_sample = samples_pool[i % X.shape[0]]
        l =  lengths[i]
        d =  dilations[i]
        
        if invariances[i][1]:
            v = sample_shapelet_phase_invariant(X[id_sample,0],l,d)
        else:
            v = sample_shapelet(X[id_sample,0], l, d)
        
        if invariances[i][0]:
            v = (v - np.mean(v))/(np.std(v) + 1e-8)
        
        id_test = np.random.choice(np.where(y == y[id_sample])[0])
        
        x_dist = dist_vect(
            X[id_test, 0], v, l, d, invariances[i]
        )
        values[i, :l] = v
        threshold[i] = np.random.uniform(
            np.percentile(x_dist, p_min), np.percentile(x_dist, p_max)
        )
        
    return values, lengths, dilations, threshold, invariances


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

        if _dist < threshold:
            _n_match += 1

    return _min, np.float64(_argmin), np.float64(_n_match)


class MINDLESS():
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
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11], n_jobs=-1,
                 p_norm=np.array([0.5,0.5,0.5,0.5]), percentiles=[5, 10], random_state=None):
        self.n_shapelets = n_shapelets
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.p_norm = p_norm
        self.percentiles = percentiles
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

    def _get_shp_params(self, id_shp):
        #values, length, dilation, padding, range
        return (self.values_[id_shp], self.length_[id_shp],
                self.dilation_[id_shp], self.threshold_[id_shp],
                self.normalize_[id_shp])

    def visualise_one_shapelet(self, id_shp, X, y, target_class, figsize=(15, 10)):
        """
        A function used to generate a visualization of a shapelet. The fit 
        function must be called before to generate shapelets, then, by giving
        the identifier (between [0, n_shapelets-1]), a visualization of the
        shapelet is produced, giving boxplot of the features it generate on 
        passed data, and a visualization on two randomly choosed samples
        between the target class and the other classes.
        
        Parameters
        ----------
        id_shp : int
            Identifier of the shapelet, must be between 0 and n_shapelets-1
        X : array, shape=(n_samples, n_features, n_timestamps)
            Input time series.
        y : array, shape=(n_samples)
            Class of the input time series.
        target_class : int
            Class to visualize. Will influence boxplot generation and sample
            choice.
        figsize : tuple, optional
            A tuple of int indicating the size of the generated figure.
            The default is (15, 10).

        Returns
        -------
        None.

        """
        # For visualisation, if argmin is important, draw a bar on x axis
        # If min, highligh on series (red)
        # If #match, hihgligh all parts which match on series (blue)
        sns.set()
        sns.set_context('talk')
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=figsize)
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
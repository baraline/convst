# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:17:07 2021

@author: a694772
"""


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from convst.utils.dataset_utils import load_sktime_dataset_split
import time
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
from joblib import Parallel
from sklearn.utils.fixes import delayed
import heapq
from numba.typed import List
from matplotlib import pyplot as plt

def yield_resamples(n_timestamps, max_len, min_len, y, p_norm, n_samples, batch_size):
    n_classes = np.unique(y).shape[0]
    if isinstance(max_len, float):
        max_len = int(n_timestamps*max_len)
    if isinstance(min_len, float):
        min_len = int(n_timestamps*min_len)
        
    possible_lengths = np.arange(min_len,max_len+1)
    powers = np.log2(np.floor_divide(n_timestamps - 1, possible_lengths - 1))
    max_dilation = np.floor(np.power(2, powers)).astype(np.int64)
    # Can we hash all functions to keep track of those generated, and give a shuffle of all possible hash instead while true ?

    for i_batch in range(batch_size):
        i_l = np.random.choice(possible_lengths.shape[0])
        l = possible_lengths[i_l]
        d = np.random.choice(max_dilation[i_l]) + 1
        norm = np.random.rand() < p_norm
        n_select = np.random.choice(range(2, min(6, n_classes+1)))
        selected_classes = np.random.choice(
            range(n_classes), size=n_select, replace=False)
        X_index = []
        for i_cls in selected_classes:
            idx_class = np.where(y == i_cls)[0]
            if idx_class.shape[0]>=n_samples:
                X_index.extend(idx_class)
            else:
                X_index.extend(np.random.choice(idx_class, size=n_samples, replace=False))
        yield l, d, norm, np.asarray(X_index, dtype=int)


@njit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _gini_split(y_node, y_left, y_right, class_weights):
    """

    Parameters
    ----------
    y_node : array, shape=(n_samples)
        Classes of the life cycles in the current node.
    y_left : Tarray, shape=(n_samples_left)
        Classes of the life cycles in the left node.
    y_right : array, shape=(n_samples_right)
        Classes of the life cycles in the right node.
    class_weights : array, shape=(n_classes)
        Weight of each class.
    Returns
    -------
    gain : float
        Information gain for the given split
    prop_left : float
        (Weighted) Proportion of n_samples in the left node.
    prop_right : float
        (Weighted) Proportion of n_samples in the right node.
    """
    n_targets = class_weights.shape[0]
    n_classes_node = np.zeros(n_targets)
    n_classes_left = np.zeros(n_targets)
    n_classes_right = np.zeros(n_targets)
    total_sum_node = 0
    total_sum_left = 0
    total_sum_right = 0
    for k in prange(n_targets):
        n_classes_node[k] = (y_node == k).sum() * class_weights[k]
        n_classes_left[k] = (y_left == k).sum() * class_weights[k]
        n_classes_right[k] = (y_right == k).sum() * class_weights[k]
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


@njit(cache=True, nogil=True, fastmath=True)
def compute_gain(th, D, y, n_classes):
    ft = np.zeros(D.shape[0])
    for i in prange(D.shape[0]):
        for j in prange(D.shape[1]):
            if D[i, j] <= th:
                ft[i] += 1
    u_ft = np.unique(ft)
    gain = 0
    cw = y.shape[0] / (n_classes * np.bincount(y))
    for i in prange(u_ft.shape[0]-1):
        val_split = (u_ft[i] + u_ft[i+1])/2
        _g = _gini_split(y, y[ft <= val_split], y[ft > val_split], cw)
        if _g > gain:
            gain = _g
    return gain

@njit(cache=True, nogil=True)
def get_scores(D, Dp, y, iX, n_classes):

    global_score = 0
    local_score = 0
    # percentage of reduction from previous step
    pointwise_diff = 1 - (D+1e-8)/(Dp+1e-8)

    per_class_diff = np.zeros(n_classes)
    ix_range = np.ones(pointwise_diff.shape[0], dtype=np.bool_)
    ix_range[iX] = False
    mask = np.ones(n_classes)
    mask[y[iX]] = False
    for i in prange(n_classes):
        per_class_diff[i] = pointwise_diff[(y == i) & (ix_range)].mean()
    local_score = np.zeros(n_classes)
    for i in prange(n_classes):
        _s = per_class_diff[i] + per_class_diff[y[iX]]
        _d = abs(per_class_diff[i] - per_class_diff[y[iX]])
        if _s > 0:
            local_score[i] = _d/_s
    global_score = pointwise_diff.mean()

    return global_score, local_score.max()

@njit(cache=True, fastmath=True, nogil=True)
def get_all_subs(X, n_samples, dist_vect_shape, l, d, p_norm):
    all_subs = np.zeros((n_samples, dist_vect_shape, l))
    CEs = np.zeros((n_samples, dist_vect_shape))
    for i in prange(n_samples):
        all_subs[i] = generate_strides_1D(X[i,0],l,d)
        for j in prange(dist_vect_shape):
            if p_norm:
                _std = all_subs[i,j].std()
                if _std != 0:
                    all_subs[i,j] = (all_subs[i,j] - all_subs[i,j].mean())/_std
                else:
                    all_subs[i,j] = np.zeros(l)
            CEs[i,j] = _CE(all_subs[i,j], l)
    return all_subs, CEs

@njit(cache=True, fastmath=True)
def _CE(X, l):
    s = 0
    for i in prange(l-1):
        s += (X[i] - X[i+1])**2
    return np.sqrt(s)


@njit(cache=True, nogil=True)
def get_threshold(candidate_distances,pmax, y, _class, n_classes):
    best_gain = 0
    best_th = 0
    for perc in range(0, pmax):
        th = np.percentile(candidate_distances[y == _class], perc)
        gain = compute_gain(
            th, candidate_distances, (y == _class).astype(np.int64), n_classes
        )
        if best_gain < gain:
            best_gain = gain
            best_th = th
    return best_th

@njit(cache=True, nogil=True)
def get_losses(pointwise_loss, y, n_classes):
    _sum = 0
    _sum_classes = np.zeros(n_classes)

    n_samples, d_vect_shape = pointwise_loss.shape
    for i in prange(n_samples):
        _s = 0
        for j in prange(d_vect_shape):
            _s += pointwise_loss[i,j]
        _sum += _s
        _sum_classes[y[i]] += _s
    for i in prange(n_classes): 
        _sum_classes[i] /= _sum
    return _sum, _sum_classes
        

@njit(cache=True, nogil=True)
def generate_shapelets_from_resample(X, y, l, d, p_norm, e, std_threshold ,pmax):
    n_samples, n_features, n_timestamps = X.shape

    classes = np.unique(y)
    n_classes = classes.shape[0]
    for i in prange(n_classes):
        y[y == classes[i]] = i
    is_first = True
    dist_vect_shape = n_timestamps - (l-1)*d
    look_around_mask = np.int64(np.floor(0.25*l)+1)
    min_dists = np.zeros((n_samples, dist_vect_shape)) + np.inf
    sampling_loc = np.random.choice(n_samples*dist_vect_shape)
    continue_search = True
    #dd = []
    #ddd = []
    all_subs, CEs = get_all_subs(X, n_samples, dist_vect_shape, l, d, p_norm)

    while continue_search:
        candidate_distances = np.zeros((n_samples, dist_vect_shape))
        i_X = sampling_loc // dist_vect_shape
        shp_class = y[i_X]
        i_T = sampling_loc % dist_vect_shape
        shp_val = [0.]*l
        # needed to keep list type for heap insertion
        for i_l in prange(l):
            shp_val[i_l] = all_subs[i_X, i_T, i_l]
                
        CE_shp = _CE(shp_val, l)
        for i_l in prange(look_around_mask):
            idx = i_T-(i_l*d)
            if idx >= 0:
                min_dists[i_X, idx] = 0
            idx2 = i_T+(i_l*d)
            if idx2 < dist_vect_shape:    
                min_dists[i_X, idx2] = 0
        
        for i in prange(n_samples):
            for j in prange(dist_vect_shape):
                for k in prange(l):
                   candidate_distances[i,j] += (shp_val[k] - all_subs[i,j,k])**2
                _min_ce = min(CE_shp,CEs[i,j])
                if _min_ce > 0:
                    candidate_distances[i,j] *= (max(CE_shp,CEs[i,j]) / min(CE_shp,CEs[i,j]))
                else:
                    candidate_distances[i,j] *= max(CE_shp,CEs[i,j])
                if min_dists[i,j] > candidate_distances[i,j]:
                    min_dists[i,j] = candidate_distances[i,j]
           
        global_sum, classes_sum = get_losses(candidate_distances, y, n_classes)
        if is_first:
            th = get_threshold(candidate_distances, pmax, y, shp_class, n_classes)
            values_heap = [shp_val]
            threshold_heap = [th]
            is_first = False
            initial_loss = np.copy(candidate_distances)
            initial_loss_sum, _ = get_losses(initial_loss, y, n_classes)
        else:
            std_class = np.std(classes_sum)
            current_loss_sum, _ = get_losses(min_dists, y, n_classes)
            if current_loss_sum/initial_loss_sum > e:
                if std_class > std_threshold:
                    th = get_threshold(candidate_distances, pmax, y, shp_class, n_classes)
                    heapq.heappush(values_heap, shp_val)
                    heapq.heappush(threshold_heap, th)
            else:
                continue_search = False
        
        
        x_loc, y_loc = np.where((min_dists >= np.percentile(min_dists, 99)))
        sampling_loc = np.random.choice(x_loc.shape[0])
        sampling_loc = x_loc[sampling_loc]*dist_vect_shape + y_loc[sampling_loc]
        
    """
    d_class = np.asarray(dd)
    dd_class = np.asarray(ddd)
    fig, ax =plt.subplots(ncols=2,nrows=2, figsize=(10,10))
    for i in range(n_classes):
        ax[0,0].plot(d_class[:,y==i].sum(axis=1), alpha=0.5)
        ax[0,1].plot(dd_class[:,y==i].sum(axis=1), alpha=0.5)
        ax[1,0].plot(np.diff(d_class.sum(axis=1)), alpha=0.5)
        ax[1,1].plot(np.diff(dd_class.sum(axis=1)), alpha=0.5)
    plt.show()
    """
    return values_heap, threshold_heap, [l]*len(values_heap), [d]*len(values_heap), [p_norm]*len(values_heap)


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
                                     ], values[i_shp, :lengths[i_shp]], threshold[i_shp]
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

    # For each step of the moving window in the shapelet distance
    for i in range(n_candidates):
        _dist = 0
        # For each value of the shapelet
        for j in prange(length):
            _dist += (x[i, j] - values[j])**2

        if _dist < _min:
            _min = _dist
            _argmin = i

        if _dist <= threshold:
            _n_match += 1

    return _min, np.float64(_argmin), np.float64(_n_match)


class IR_DST(BaseEstimator, TransformerMixin):
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

    def __init__(self, contract_max_n_shapelet_samples=10_000, n_per_class_resample=2,
                 time_limit_in_minutes=1, pmax=20, std_threshold=0.1, e=0.01, min_len=0.01,
                 max_len=0.1, p_norm=0.8, n_jobs=1, random_state=None):
        self.random_state = random_state
        self.p_norm = p_norm
        self.min_len = min_len
        self.max_len = max_len
        self.e = e
        self.std_threshold = std_threshold
        self.pmax = pmax
        self.n_jobs = check_n_jobs(n_jobs)
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples
        self.n_per_class_resample = n_per_class_resample
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

        # Generate the shapelets
        n_shapelets_extracted = 0
        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        fit_time = 0
        self.values_ = []
        self.length_ = []
        self.dilation_ = []
        self.normalize_ = []
        self.threshold_ = []
        if time_limit > 0:
            while (
                fit_time < time_limit
                and n_shapelets_extracted < self.contract_max_n_shapelet_samples
            ):
                results = Parallel(
                    n_jobs=self.n_jobs, prefer='threads')(
                    delayed(generate_shapelets_from_resample)(
                        X[X_index], y[X_index], l, d, norm,
                        self.e, self.std_threshold, self.pmax
                    )
                    for l, d, norm, X_index in yield_resamples(
                        n_timestamps, self.max_len, self.min_len, y, self.p_norm,
                        self.n_per_class_resample, self.n_jobs
                    )
                )
                for i_res in range(self.n_jobs):
                    self.values_ += results[i_res][0]
                    self.threshold_ += results[i_res][1]
                    self.length_ += results[i_res][2]
                    self.dilation_ += results[i_res][3]
                    self.normalize_ += results[i_res][4]
                n_shapelets_extracted = len(self.values_)
                #print(n_shapelets_extracted)
                fit_time = time.time() - start_time

        vals = np.zeros((len(self.values_,), max(self.length_)))
        for i in range(len(self.values_)):
            vals[i, :self.length_[i]] = self.values_[i]

        self.values_ = vals
        self.length_ = np.asarray(self.length_)
        self.dilation_ = np.asarray(self.dilation_)
        self.normalize_ = np.asarray(self.normalize_).astype(np.int64)
        self.threshold_ = np.asarray(self.threshold_)

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

# In[]:
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(
    'PigAirwayPressure')
print(X_train.shape)
print(X_test.shape)
print(np.bincount(y_train))
# In[]:
e = 0.01
std_th = 0.05
n_sp = 1
a = IR_DST(
    n_jobs=1, e=e, std_threshold=std_th, min_len=0.01, 
    max_len=0.02, pmax=10, time_limit_in_minutes=3,
    n_per_class_resample=n_sp, contract_max_n_shapelet_samples=10000
).fit(X_train, y_train)

classifier = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeClassifierCV()
)

at = a.transform(X_train)
att = a.transform(X_test)

#c = RotationForest(n_jobs=3, n_estimators=60, min_group=3, max_group=3).fit(at, y_train)
# print(accuracy_score(c.predict(att),y_test))
classifier.fit(at, y_train)
print("Score e = {}, std = {}, n_sample = {}".format(e, std_th, n_sp))
print(classifier.score(att, y_test))
print("--------------------")
# In[]:
r = R_DST_Ridge(n_jobs=3).fit(X_train, y_train)
print(r.score((X_test), y_test))

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:17:07 2021

@author: a694772
"""
from numba import njit, prange
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from convst.utils.checks_utils import check_array_3D, check_array_1D
from convst.utils.shapelets_utils import generate_strides_1D
from numba import set_num_threads
from matplotlib import pyplot as plt

@njit(fastmath=True, cache=True, error_model='numpy')
def compute_shapelet_dist_vector(x, values, length, dilation):
    c = generate_strides_1D(x, length, dilation)
    x_conv = np.empty(c.shape[0])
    for i in range(x_conv.shape[0]):
        s = 0
        for j in range(length):
            s += abs(c[i,j] - values[j])
        x_conv[i] = s
    return x_conv


@njit(cache=True, parallel=True)
def _init_random_shapelet_params(n_shapelets, shapelet_sizes, n_timestamps):
    # Lengths of the shapelets
    lengths = np.random.choice(shapelet_sizes, size=n_shapelets).astype(np.int64)

    # Dilations
    upper_bounds = np.log2(np.floor_divide(n_timestamps - 1, lengths - 1))
    powers = np.empty(n_shapelets)
    for i in prange(n_shapelets):
        powers[i] = np.random.uniform(0, upper_bounds[i])
    dilations = np.floor(np.power(2, powers)).astype(np.int64)
    
    # Init ranges array
    ranges = np.zeros(n_shapelets,dtype=np.float64)
    
    # Init values array
    values = np.zeros((n_shapelets, np.int64(np.max(shapelet_sizes))),
                      dtype=np.float64)
    
    return values, lengths, dilations, ranges

@njit(cache=True, fastmath=True, error_model='numpy')
def _get_subsequence(X, i_start, l, d):
    v = np.empty(l,dtype=np.float64)
    _idx = i_start
    for j in range(l):
        v[j] = X[_idx]
        _idx += d
    return v

@njit(cache=True, parallel=True)
def generate_shapelet(X, y, n_shapelets, shapelet_sizes, seed, p_min, p_max):
    n_samples, n_features, n_timestamps = X.shape

    # Fix the random see
    np.random.seed(seed)

    values, lengths, dilations, ranges =  _init_random_shapelet_params(
        n_shapelets, shapelet_sizes, n_timestamps
    )
    
    samples_pool = np.arange(X.shape[0]).astype(np.int64)
    np.random.shuffle(samples_pool)
    # For Values, draw from random uniform (0,n_samples*(n_ts-(l-1)*d))
    # for each l,d combinations. Then take by index the values instead
    # of generating strides.
    for i in prange(n_shapelets):
        id_sample = samples_pool[i%X.shape[0]]
        index = np.int64(np.random.choice(
            n_timestamps - (lengths[i]-1)*dilations[i]
        ))
        v = _get_subsequence(
            X[id_sample,0], index, lengths[i],dilations[i]
        )
        
        values[i, :lengths[i]] = v
        
        id_test=np.random.choice(np.where(y==y[id_sample])[0])
        
        x_dist = compute_shapelet_dist_vector(
            X[id_test,0], values[i], lengths[i], dilations[i]
        )
        ranges[i] = np.random.uniform(
            np.percentile(x_dist,p_min), np.percentile(x_dist,p_max)
        )
    
    return values, lengths, dilations, ranges


@njit(cache=True, parallel=True, fastmath=True, error_model='numpy')
def apply_all_shapelets(X, values, lengths, dilations, ranges):
    n_samples, n_ft, n_timestamps = X.shape
    n_shapelets = len(lengths)
    n_features = 3
    
    unique_lengths = np.unique(lengths)
    unique_dilations = np.unique(dilations)
    
    X_new = np.empty((n_samples, n_features * n_shapelets), dtype=np.float64)
    for index_l in range(unique_lengths.shape[0]):
        l=unique_lengths[index_l]
        for index_d in prange(unique_dilations.shape[0]):
            d = unique_dilations[index_d]
            ix_shapelets = np.where((lengths==l)&(dilations==d))[0]
            if len(ix_shapelets)>0:
                for i in prange(n_samples):
                    strides = generate_strides_1D(X[i,0], l, d)
                    for j in prange(len(ix_shapelets)):
                        i_shp = ix_shapelets[j]
                        X_new[i, (n_features * i_shp):(n_features * i_shp + n_features)] = apply_one_shapelet_one_sample(
                            strides, values[i_shp], ranges[i_shp]
                        )
    return X_new

@njit(fastmath=True, cache=True)
def apply_one_shapelet_one_sample(x, values, r):
    n_candidates, length = x.shape

    _n_match = 0
    _min = 1e+100
    _argmin = 0
    
    #For each step of the moving window in the shapelet distance
    for i in range(n_candidates):        
        _dist = 0
        #For each value of the shapelet
        for j in prange(length):
            _dist += abs(x[i,j] - values[j])
            
        if _dist < _min:
            _min = _dist
            _argmin = i

        if _dist < r:
            _n_match += 1

    return _min, _argmin/n_candidates, _n_match/n_candidates

class R_DST_NN(BaseEstimator, TransformerMixin):
    def __init__(self, n_shapelets=10000, shapelet_sizes=[11],
                 random_state=None, percentiles=[5,10]):
        self.n_shapelets = n_shapelets
        self.shapelet_sizes = np.asarray(shapelet_sizes)
        self.random_state = random_state
        self.percentiles = percentiles
        
    def fit(self, X, y=None):
        X = check_array_3D(X).astype(np.float64)
        n_samples, n_features, n_timestamps = X.shape
        if self.shapelet_sizes.dtype == float:
            self.shapelet_sizes = np.floor(n_timestamps*self.shapelet_sizes)
        for i in range(self.shapelet_sizes.shape[0]):
            if self.shapelet_sizes[i] < 5:
                self.shapelet_sizes[i] = 5
        shapelet_sizes, seed = self._check_params(n_timestamps)
        # Generate the shapelets
  
        values, lengths, dilations, ranges = generate_shapelet(
            X, y, self.n_shapelets, shapelet_sizes, seed,
            self.percentiles[0], self.percentiles[1]
        )
        self.values_ = values
        self.length_ = lengths
        self.dilation_ = dilations
        self.ranges_ = ranges

        return self

    def transform(self, X):
        X = check_array_3D(X).astype(np.float64)
        check_is_fitted(self, ['values_', 'length_', 'dilation_', 'ranges_'])
        X_new = apply_all_shapelets(
            X, self.values_, self.length_, self.dilation_, self.ranges_
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
                self.dilation_[id_shp], self.ranges_[id_shp],
                self.normalize_[id_shp])

    def visualise_one_shapelet(self, id_shp, X, y, target_class):
        # For visualisation, if argmin is important, draw a bar on x axis
        # If min, highligh on series (red)
        # If #match, hihgligh all parts which match on series (blue)
        sns.set()
        sns.set_context('talk')
        n_classes = np.unique(y).shape[0]
        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15,15))
        values, length, dilation, r, norm = self._get_shp_params(id_shp)
        X_new = np.zeros((X.shape[0], 3))
        yc = (y==target_class).astype(int)
        for i in range(X.shape[0]):
            x_dist = compute_shapelet_dist_vector(X[i,0], values, length, dilation, norm)
            X_new[i,0] = np.min(x_dist)
            X_new[i,1] = np.argmin(x_dist)
            X_new[i,2] = np.mean(x_dist < r)
    
        sns.boxplot(x=yc, y=X_new[:,0], ax=ax[0,0])
        sns.boxplot(x=yc, y=X_new[:,1], ax=ax[1,0])
        sns.boxplot(x=yc, y=X_new[:,2], ax=ax[2,0])
        
        i0 = np.random.choice(np.where(yc==0)[0])
        i1 = np.random.choice(np.where(yc==1)[0])
        ax[0,1].plot(X[i1, 0], c='C1', alpha=0.75, label='sample of class {}'.format(target_class))
        ax[0,1].plot(X[i0, 0], c='C0', alpha=0.75, label='sample of class {}'.format(y[i0]))
        ax[1,1].plot(X[i1, 0], c='C1', alpha=0.75, label='sample of class {}'.format(target_class))
        ax[1,1].plot(X[i0, 0], c='C0', alpha=0.75, label='sample of class {}'.format(y[i0]))
        ax[2,1].plot(compute_shapelet_dist_vector(X[i1,0], values, length, dilation, norm), c='C1', alpha=0.75, label='distance vector of sample of class {}'.format(target_class))
        ax[2,1].plot(compute_shapelet_dist_vector(X[i0,0], values, length, dilation, norm), c='C0', alpha=0.75, label='distance vector of sample of class {}'.format(y[i0]))
        ax[0,1].legend()
        ax[1,1].legend()
        ax[2,1].legend()
        ax[0,0].set_xticks([0,1])
        ax[0,0].set_xticklabels(['other classes','class {}'.format(target_class)])
        ax[1,0].set_xticks([0,1])
        ax[1,0].set_xticklabels(['other classes','class {}'.format(target_class)])
        ax[2,0].set_xticks([0,1])
        ax[2,0].set_xticklabels(['other classes','class {}'.format(target_class)])
        
        ix_start = X_new[i0, 1]
        if norm == 1:
            ix = np.zeros(length)
            for i in range(length):
                ix[i] = ix_start + (i*dilation)
            ix = ix.astype(int)
            v = values[:length] * X[i0, 0, ix].std() + X[i0, 0, ix].mean()
        else:
            v = values[:length]
        ax[0,1].scatter(ix, v, c='C0', alpha=0.75)
        
        ix_start = X_new[i1, 1]
        if norm == 1:
            ix = np.zeros(length)
            for i in range(length):
                ix[i] = ix_start + (i*dilation)
            ix = ix.astype(int)
            v = values[:length] * X[i1, 0, ix].std() + X[i1, 0, ix].mean()
        else:
            v = values[:length]
        ax[0,1].scatter(ix, v, c='C1', alpha=0.75)
        
        ax[1,1].scatter(int(X_new[i0,1]), X[i0,0, int(X_new[i0,1])], c='C0', alpha=0.75)
        ax[1,1].scatter(int(X_new[i1,1]), X[i1,0, int(X_new[i1,1])], c='C1', alpha=0.75)
        
        ax[2,1].axhline(r, c='C2', linestyle='--')
        """
        ax[i,1].set_xticks(np.arange(length), rotation=70)
        ax[i,1].set_xticklabels(np.arange(length)*dilation, rotation=70)
        ax[i,2].plot(x_conv, color = 'C'+str(i))
        ax[i,2].axhline(r)
        """    
        ax[0,0].set_title("Min")
        ax[0,1].set_title("0 : {}; 1 : {}".format(str(X_new[i0,0])[0:5],str(X_new[i1,0])[0:5]))
        ax[1,0].set_title("Argmin")
        ax[1,1].set_title("0 : {}; 1 : {}".format(str(X_new[i0,1])[0:5],str(X_new[i1,1])[0:5]))
        ax[2,0].set_title("# Match")
        ax[2,1].set_title("0 : {}; 1 : {}".format(str(X_new[i0,2])[0:5],str(X_new[i1,2])[0:5]))
        
        fig.suptitle("Shapelet l={}, d={}, n={}".format(length,dilation,norm))
        plt.show()
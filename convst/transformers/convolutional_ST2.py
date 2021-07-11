# -*- coding: utf-8 -*-

import numpy as np
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin

from numba import njit, prange
from numba import set_num_threads

from scipy.spatial.distance import cdist

from convst.transformers import MiniRocket, ForestSplitter
from convst.utils import check_array_3D
from convst.utils import generate_strides_2D


class ConvolutionalShapeletTransformer_onlyleaves(BaseEstimator, TransformerMixin):
    """
    Implementation of univariate Convolutional Shapelet Transform (CST).
    For details and explanation on the algorithm, users are referred to [1]_:

    Parameters
    ----------
    P : int, optional
        Percentile used in the candidate selection. The default is 80.
    n_trees : int, optional
        Number of trees on which to extract the nodes. The default is 100.
    max_ft : float or int, optional
        Percentage of features to consider at each node. The default is 1.0.
    n_bins : int, optional
        Number of bins used to discretize the shapelets. The default is 11.
    leaves_only : boolean, optional
        Wheter or not to only use node with at least one leaf in direct childs
    class_weights : str or dict, optional
        Wheter or not to balance classes. The default is 'balanced'. None will
        apply no balancing. This is passed to the random forest in fit.
    verbose : int, optional
        Control the level of verbosity output. The default is 0.
    n_jobs : int, optional
        Number of parallel jobs to execute. The default is 3.
    ccp_alpha : float, optional
        Post pruning coefficient applied to trees. The default is 0.0.
    random_state : int, optional
        Value of the random state for trees. The default is None.


    Attributes
    ----------
    shapelets : array, shape=(n_shapelets, kernel_length)
        Shapelets will be stored in this array after calling the fit method.
    dilation : array, shape=(n_shapelets)
        The dilation parameter of each shapelets will be stored in this
        array after calling the fit method.


    Notes
    -----
    .. [1] Guillaume A. et al, "Convolutional Shapelet Transform:
        A new approach of time series shapelets" (2021)

    """

    def __init__(self,  P=80, n_trees=100, max_ft=1.0, class_weights='balanced',
                 verbose=0, n_bins=11, n_jobs=3, leaves_only=True,
                 ccp_alpha=0.0, random_state=None):
        self.verbose = verbose
        self.ccp_alpha = ccp_alpha
        self.P = P
        self.n_trees = n_trees
        self.leaves_only = leaves_only
        self.class_weights = class_weights
        self.max_ft = max_ft
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.random_state = random_state
        set_num_threads(n_jobs)

    def _log(self, message):
        if self.verbose > 0:
            print(message)

    def _log2(self, message):
        if self.verbose > 1:
            print(message)

    def fit(self, X, y):
        """
        Extract shapelets from the input. First generate kernels and
        extract tree nodes, then for each node, extract shapelets representing
        the split made by the node.

        Parameters
        ----------
        X : array, shape=(n_samples, 1, n_timestamps)
            An univariate time series dataset
        y : array, shape=(n_samples)
            The class associated with the input time series.

        Returns
        -------
        self
            Fitted instance of CST


        """
        X = check_array_3D(X, is_univariate=True)
        ft, L, dils = self._init_rocket(X, y)
        id_X_leaves, kernel_id = self._generate_nodes(ft, y)
        # TODO : Deleting/collecting after numba calls seems to help with memory consumption peaks, to study
        del ft
        gc.collect()

        i_unique = is_unique(id_X_leaves, kernel_id.reshape(-1, 1))
        id_X_leaves = id_X_leaves[i_unique]
        kernel_id = kernel_id[i_unique]
        u_kernels = np.unique(kernel_id)
        L = L[:, u_kernels, :]
        dils = dils[u_kernels]

        del i_unique, u_kernels
        gc.collect()
        self._log("Extracting shapelets from {} splits ...".format(
            id_X_leaves.shape[0]))
        shp, d = _extract_candidates(
            X, id_X_leaves, kernel_id, L, dils, self.P)
        del id_X_leaves
        del kernel_id
        gc.collect()

        self._log("Discretizing {} shapelets ...".format(shp.shape[0]))
        n_shapelets = 0
        shps = {}
        for dil in np.unique(d):
            candidates = self._discretize(shp[d == dil])
            n_shapelets += candidates.shape[0]
            shps.update({dil: candidates})
        self.shapelets = np.zeros((n_shapelets, 9), dtype=np.float32)
        self.dilation = np.zeros(n_shapelets, dtype=np.uint32)
        prec = 0
        for key in shps:
            self.shapelets[prec:prec+shps[key].shape[0]] += shps[key]
            self.dilation[prec:prec+shps[key].shape[0]] += key
            prec += shps[key].shape[0]
        del shp
        del shps
        del d
        gc.collect()
        return self

    def _discretize(self, candidates):
        """
        Discretize shapelet candidates by changing their values to the mean
        of the two bin edges in which they are contained. Remove duplicates before
        returning the new shapelets

        Parameters
        ----------
        candidates : array, shape=(n_candidates, kernel_length)
            Shapelet candidates extracted from input time series

        Returns
        -------
        candidates : array, shape=(n_unique_candidates, kernel_length)
            The discretized unique candidates

        """
        candidates = (candidates - candidates.mean(axis=-1, keepdims=True)) / (
            candidates.std(axis=-1, keepdims=True) + 1e-8)
        if not np.all(candidates.reshape(-1, 1) == candidates.reshape(-1, 1)[0]):
            kbd = KBinsDiscretizer(n_bins=self.n_bins,
                                   strategy='uniform',
                                   dtype=np.float32).fit(
                                       candidates.reshape(-1, 1))
            candidates = np.unique(kbd.inverse_transform(
                kbd.transform(candidates.reshape(-1, 1))).reshape(-1, 9), axis=0)
        else:
            candidates = np.unique(candidates, axis=0)
        return candidates

    def _init_rocket(self, X, y):
        """
        Create the kernels by using mini rocket.

        Parameters
        ----------
        X : array, shape=(n_samples, 1, n_timestamps)
            An univariate time series dataset
        y : array, shape=(n_samples)
            The class associated with the input time series.

        Returns
        -------
        ft : array, shape=(n_samples, n_kernels)
            The ppv features extracted from the kernels
        L : array, shape(n_samples, n_kernels, n_timestamps)
            Counts of how many time each point of each sample was used in
            convolution operations of a kernel that resulted in a positive value
        dils : array, shape=(n_kernels)
            The dilation parameter used by the kernels

        """
        self._log("Initializing kernels ...")
        m = MiniRocket(random_state=self.random_state).fit(X)
        self._log("Computing convolutions ...")
        ft, L = m.transform(X, return_locs=True)
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=np.int32)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] += dilations[i]
            n += 84*num_features_per_dilation[i]
        return ft, L, dils

    def _generate_nodes(self, ft, y):
        """
        Fit a RandomForest and extract the input and output data from each
        node in the forest.

        Parameters
        ----------
        ft : array, shape=(n_samples, n_kernels)
            The ppv features extracted from the kernels
        y : array, shape=(n_samples)
            The class associated with the input time series.

        Returns
        -------
        id_X_split : array, shape=(n_leaves, n_samples)
            Indicate for each leaves of the forest which input
            time series was used as input in the parent node.
            The encoding is -1 for a non-used sample, 0 for a sample
            that ended in the leaf, 1 for the other samples.
        kernel_id : array, shape=(n_leaves)
            Indicate the kernel which was used to generate the ppv feature
            used in the split of the parent node of a leaf.


        """
        self._log("Fitting forest with {} kernels ...".format(ft.shape[1]))
        
        rf = RandomForestClassifier(n_estimators=self.n_trees,
                                    max_features=self.max_ft,
                                    class_weight=self.class_weights,
                                    ccp_alpha=self.ccp_alpha,
                                    n_jobs=self.n_jobs,
                                    random_state=self.random_state)

        rf.fit(ft, y)
        
        fs = ForestSplitter(rf, leaves_only=self.leaves_only)
        id_X_leaves, kernel_id = fs.fit_transform(ft)
        print(id_X_leaves.shape)
        return id_X_leaves, kernel_id
    
    def transform(self, X, return_inverse=False):
        """
        Transform the input data with the shapelet transform to obtain a distance
        matrix.

        Parameters
        ----------
        X : array, shape=(n_samples, 1, n_timestamps)
            An univariate time series dataset

        return_inverse : bool, optional
            Wheter or not to return the inverse of the distances.
            The default is False.

        Returns
        -------
        array, shape=(n_samples, n_shapelets)
            The shapelet distance between input time series and shapelets

        """
        self._check_is_fited()
        X = check_array_3D(X)
        distances = np.zeros(
            (X.shape[0], self.shapelets.shape[0]), dtype=np.float32)
        prev = 0
        self._log("Perfoming transform ...")
        for dilation in np.unique(self.dilation):
            X_strides = generate_strides_2D(X[:, 0, :], 9, dilation)
            X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
                X_strides.std(axis=-1, keepdims=True) + 1e-8)
            mask = self.dilation == dilation
            self._log2("Transforming for dilation {} with {} shapelets ...".format(
                dilation, self.shapelets[mask].shape[0]))

            d = np.asarray([cdist(X_strides[j], self.shapelets[mask],
                                  metric='sqeuclidean').min(axis=0)
                            for j in range(X.shape[0])])
            distances[:, prev:prev+d.shape[1]] = d
            prev += d.shape[1]
        if return_inverse:
            return 1/(distances+1e-8)
        else:
            return distances

    def _check_is_fited(self):
        if any(self.__dict__[attribute] is None for attribute in ['shapelets']):
            raise AttributeError("This instance of CST was not fited, "
                                 "call fit before trying to transform data")


def is_unique(*lsts):
    """
    Stack lists supplied as parameter along axis 0 and return a boolean
    mask indicating which samples are to be kept to remove all duplicates
    from the combination of all lists.

    Parameters
    ----------
    *lsts : arrays, shape=(n_samples, _)
        An ensemble of arrays with the same number of samples, but which
        can have a different amount of features.

    Returns
    -------
    out : array, shape=(n_samples)
        Boolean mask in which True values indicate the samples to be kept
        so the ensemble of arrays do not contains duplicates.

    """
    arr = np.hstack(lsts)
    _, ind = np.unique(arr, axis=0, return_index=True)
    out = np.zeros(shape=arr.shape[0], dtype=bool)
    out[ind] = True
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _extract_candidates(X, id_X_leaves, kernel_id, L, dils, P):
    # x_mask record for each leaf all tuples
    # (sample, timestamp) that were extracted from a region
    n_timestamps = X.shape[2]
    x_mask = _extract_location_mask(id_X_leaves, kernel_id,
                                    dils, L, P, n_timestamps)

    n_candidates = np.sum(x_mask)
    candidates = np.zeros((n_candidates, 9), dtype=np.float32)
    candidates_dil = np.zeros((n_candidates), dtype=np.int32)

    # How much candidate are to be extracted from each leaf:
    per_leaf_id = np.zeros((id_X_leaves.shape[0]+1), dtype=np.int32)
    for i_leaf in prange(id_X_leaves.shape[0]):
        per_leaf_id[i_leaf+1] = np.sum(x_mask[i_leaf])
    per_leaf_id = per_leaf_id.cumsum()

    unique_kernels = np.unique(kernel_id)
    for i_leaf in prange(id_X_leaves.shape[0]):
        k_id = kernel_id[i_leaf]
        dil = dils[np.where(unique_kernels == k_id)[0][0]]

        candidates_dil[per_leaf_id[i_leaf]:per_leaf_id[i_leaf+1]] += dil

        indexes = np.where(x_mask[i_leaf])[0]

        x_indexes = indexes // n_timestamps
        l_indexes = indexes % n_timestamps
        for i_candidate in prange(x_indexes.shape[0]):
            for j in prange(9):
                candidates[per_leaf_id[i_leaf]+i_candidate, j] = X[
                    x_indexes[i_candidate], 0, l_indexes[i_candidate]+j*dil]
    return candidates, candidates_dil


@njit(cache=True, fastmath=True, parallel=True)
def _extract_location_mask(id_X_leaves, kernel_id, dils, L, P, n_timestamps):
    #(n_leaves, n_samples*n_timestamps)
    # Could maybe do mask per kernel and not leaf
    x_mask = np.zeros(
        (id_X_leaves.shape[0], id_X_leaves.shape[1]*n_timestamps), dtype=np.bool_)

    unique_kernels = np.unique(kernel_id)
    for i_k in prange(unique_kernels.shape[0]):
        k_id = unique_kernels[i_k]
        id_leaves = np.where(kernel_id == k_id)[0]
        dil = dils[i_k]

        Lp = generate_strides_2D(L[:, i_k, :], 9, dil).sum(axis=-1)

        for i_leaf in prange(id_leaves.shape[0]):
            id_leaf = id_leaves[i_leaf]
            x_0 = np.where(id_X_leaves[id_leaf] == 0)[0]
            y_node = id_X_leaves[id_leaf][id_X_leaves[id_leaf] >= 0]
            # To match the problem with empty trees
            if y_node.shape[0] > 0:
                c_w = y_node.shape[0] / (2 * np.bincount(y_node))

                # 0 is the id of the samples in the leaf
                # 1 is the id of the samples in the other children of the parent node
                LC = np.zeros((2, Lp.shape[1]), dtype=np.float32)
                LC[0] = c_w[0] * Lp[x_0].sum(axis=0)
                LC[1] = c_w[1] * Lp[id_X_leaves[id_leaf] == 1].sum(axis=0)

                D = LC[0] - LC[1]
                id_D = np.where(D >= np.percentile(D, P))[0]
                # A region is a set of following indexes
                regions, i_regions = _get_regions(id_D)

                for i_r in prange(i_regions.shape[0]-1):
                    region = regions[i_regions[i_r]:i_regions[i_r+1]]
                    LC_region = LC[0][region]
                    if LC_region.shape[0] > 0:
                        id_max_region = region[LC_region.argmax()]
                        x_index = np.argmax(
                            Lp[x_0, id_max_region])
                        x_mask[id_leaf, x_0[x_index]
                               * n_timestamps + id_max_region] += 1
    return x_mask


@njit(cache=True)
def _get_regions(indexes):
    regions = np.zeros((indexes.shape[0]*2), dtype=np.int32)-1
    p = 0
    for i in prange(indexes.shape[0]-1):
        regions[p] = indexes[i]
        if indexes[i] == indexes[i+1]-1:
            p += 1
        else:
            p += 2
    regions[p] = indexes[-1]
    idx = np.where(regions != -1)[0]
    return regions[idx], np.concatenate((np.array([0], dtype=np.int32), np.where(np.diff(idx) != 1)[0]+1, np.array([indexes.shape[0]], dtype=np.int32)))

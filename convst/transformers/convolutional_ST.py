# -*- coding: utf-8 -*-
"""
Created on Sat May 22 09:34:51 2021

@author: Antoine
"""

import numpy as np
import gc

from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin

from numba import njit, prange
from numba import set_num_threads

from scipy.spatial.distance import cdist

from convst.transformers.minirocket import MiniRocket
from convst.utils.checks_utils import check_array_3D
from convst.utils.shapelets_utils import generate_strides_2D


class ConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
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
    use_class_weights : boolean, optional
        Wheter or not to balance classes. The default is True.
    verbose : int, optional
        Control the level of verbosity output. The default is 0.
    n_bins : int, optional
        Number of bins used to discretize the shapelets. The default is 13.
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
    .. [1] Antoine Guillaume et al, "Convolutional Shapelet Transform: A new approach of time series shapelets" (2021)
    
    
    """

    def __init__(self,  P=80, n_trees=100, max_ft=1.0, use_class_weights=True,
                 verbose=0, n_bins=11, n_jobs=3, ccp_alpha=0.0, random_state=None):
        self.verbose = verbose
        self.ccp_alpha = ccp_alpha
        self.P = P
        self.n_trees = n_trees
        self.use_class_weights = use_class_weights
        self.max_ft = max_ft
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.random_state = random_state
        set_num_threads(n_jobs)

    def _log(self, message):
        if self.verbose > 0:
            print(message)

    def fit(self, X, y):
        """
        Extract shapelets from the input. First generate kernels and extract tree nodes,
        then for each node, extract shapelets representing the split made by the node.

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
        X_split, y_split, kernel_id = self._generate_nodes(ft, y)
        del ft
        gc.collect()

        i_unique = is_unique(X_split, y_split, kernel_id.reshape(-1, 1))
        X_split = X_split[i_unique]
        y_split = y_split[i_unique]
        kernel_id = kernel_id[i_unique]
        del i_unique
        self._log("Extracting shapelets from {} nodes ...".format(
            X_split.shape[0]))
        shp, d = _fit(X, X_split, y_split, kernel_id, L, dils, self.P)
        del X_split
        del y_split
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
            kbd = KBinsDiscretizer(n_bins=self.n_bins, strategy='uniform', dtype=np.float32).fit(
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
        m = MiniRocket().fit(X)
        self._log("Computing convolutions ...")
        ft, L = m.transform(X, return_locs=True)
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=np.uint16)
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
        X_split : array, shape=(n_nodes, n_samples)
            Indicate in a tree node which input time series was used as input 
            with a boolean mask
        y_split : array, shape=(n_nodes, n_samples)
            Indicate the partition generated by a tree node on the input samples.
            Value -1 mean that the sample is not used as input.
        kernel_id : array, shape=(n_nodes)
            Indicate the kernel which was used to generate the ppv feature
            used to split the input data of a node.

        
        """
        self._log("Fitting forest with {} kernels ...".format(ft.shape[1]))
        if self.use_class_weights:
            Forest = RandomForestClassifier(n_estimators=self.n_trees,
                                            max_features=self.max_ft,
                                            class_weight='balanced',
                                            ccp_alpha=self.ccp_alpha,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state)
        else:
            Forest = RandomForestClassifier(n_estimators=self.n_trees,
                                            max_features=self.max_ft,
                                            ccp_alpha=self.ccp_alpha,
                                            n_jobs=self.n_jobs,
                                            random_state=self.random_state)
        Forest.fit(ft, y)

        n_nodes = np.asarray([Forest.estimators_[i_dt].tree_.node_count
                              for i_dt in range(self.n_trees)])
        n_leaves = np.asarray([Forest.estimators_[i_dt].tree_.n_leaves
                               for i_dt in range(self.n_trees)])
        n_splits = (n_nodes - n_leaves).sum()
        self._log("Extracting {} forest nodes ...".format(n_nodes.sum()))
        kernel_id = np.zeros((n_splits), dtype=np.int32)
        X_split = np.zeros((n_splits, ft.shape[0]), dtype=np.bool_)
        y_split = np.zeros((n_splits, ft.shape[0]), dtype=np.int8) - 1

        i_split_nodes = np.zeros(self.n_trees+1, dtype=np.int32)
        i_split_nodes[1:] += (n_nodes - n_leaves).cumsum()

        for i_dt in range(self.n_trees):
            tree = Forest.estimators_[i_dt].tree_
            node_indicator = Forest.estimators_[i_dt].decision_path(ft)
            nodes_id = np.where(tree.feature != _tree.TREE_UNDEFINED)[0]
            kernel_id[i_split_nodes[i_dt]:i_split_nodes[i_dt+1]
                      ] += tree.feature[nodes_id]
            for i in range(nodes_id.shape[0]):
                x_index_node = node_indicator[:, nodes_id[i]].nonzero()[0]
                X_split[i_split_nodes[i_dt]+i, x_index_node] = True
                y_split[i_split_nodes[i_dt]+i, x_index_node] += (ft[
                    x_index_node, kernel_id[i_split_nodes[i_dt]+i]
                ] <= tree.threshold[nodes_id[i]]) + 1

        return X_split, y_split, kernel_id

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
        for dilation in np.unique(self.dilation):
            X_strides = generate_strides_2D(X[:, 0, :], 9, dilation)
            X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
                X_strides.std(axis=-1, keepdims=True) + 1e-8)
            mask = self.dilation == dilation
            self._log("Transforming for dilation {} with {} shapelets ...".format(
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
    arr = np.hstack(lsts)
    _, ind = np.unique(arr, axis=0, return_index=True)
    out = np.zeros(shape=arr.shape[0], dtype=bool)
    out[ind] = True
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _fit(X, X_split, y_split, kernel_id, L, dils, P):

    #x_mask record for each split and each class (2) all tuples
    #(sample, timestamp) that were extracted from a region

    x_mask = np.zeros(
        (X_split.shape[0], 2, X.shape[0]*X.shape[2]), dtype=np.bool_)
    for i_split in prange(X_split.shape[0]):
        #Take data used in the current split
        x_indexes = np.where(X_split[i_split])[0]
        k_id = kernel_id[i_split]
        y_splt = y_split[i_split][x_indexes]
        dil = dils[k_id]
        classes = np.unique(y_splt)
        n_classes = classes.shape[0]

        #Generate L prime by unweighted convolution
        Lp = generate_strides_2D(L[x_indexes, k_id, :], 9, dil).sum(axis=-1)
        #Compute class weights
        c_w = X[x_indexes].shape[0] / (n_classes * np.bincount(y_splt))

        #Compute per class score
        LC = np.zeros((n_classes, Lp.shape[1]))
        for i_class in classes:
            LC[i_class] = c_w[i_class] * Lp[y_splt == i_class].sum(axis=0)

        #Extract tuples (sample, timestamp) from each region in the differences of LC.
        for i_class in classes:
            if LC.sum() > 0:
                D = LC[i_class] - LC[(i_class+1) % 2]
                id_D = np.where(D >= np.percentile(D, P))[0]
                # A region is a set of following indexes
                regions, i_regions = _get_regions(id_D)
                for i_r in prange(i_regions.shape[0]-1):
                    region = regions[i_regions[i_r]:i_regions[i_r+1]]
                    LC_region = LC[i_class][region]
                    if LC_region.shape[0] > 0:
                        id_max_region = region[LC_region.argmax()]
                        x_index = np.argmax(
                            Lp[np.where(y_splt == i_class)[0], id_max_region])
                        x_mask[i_split, i_class, x_indexes[x_index]
                               * X.shape[2] + id_max_region] += 1

    n_candidates = np.sum(x_mask)
    candidates = np.zeros((n_candidates, 9), dtype=np.float32)
    candidates_dil = np.zeros((n_candidates), dtype=np.uint16)

    #How much candidate are to be extracted from each split
    per_split_id = np.zeros((X_split.shape[0]+1), dtype=np.int32)
    for i_split in prange(X_split.shape[0]):
        per_split_id[i_split+1] = np.sum(x_mask[i_split])
    per_split_id = per_split_id.cumsum()

    for i_split in prange(X_split.shape[0]):
        mask = X_split[i_split]
        k_id = kernel_id[i_split]
        y_splt = y_split[i_split][mask]
        dil = dils[k_id]

        candidates_dil[per_split_id[i_split]:per_split_id[i_split+1]] += dil

        classes = np.unique(y_splt)
        for i_class in classes:
            indexes = np.where(x_mask[i_split, i_class])[0]

            x_indexes = indexes//X.shape[2]
            l_indexes = indexes % X.shape[2]
            for i_candidate in prange(x_indexes.shape[0]):
                for j in prange(9):

                    candidates[per_split_id[i_split]+(i_class*(x_mask[i_split, 0].sum(
                    )))+i_candidate, j] = X[x_indexes[i_candidate], 0, l_indexes[i_candidate]+j*dil]

    return candidates, candidates_dil


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

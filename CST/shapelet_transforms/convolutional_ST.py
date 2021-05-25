# -*- coding: utf-8 -*-
"""
Created on Sat May 22 09:34:51 2021

@author: Antoine
"""

import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from CST.base_transformers.minirocket import MiniRocket
from CST.utils.checks_utils import check_array_3D
from numba import njit, prange
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.distance import cdist
from numba import set_num_threads

class ConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self,  P=90, n_trees=200, max_ft=1.0, id_ft=0, use_class_weights=True,
                 verbose=0, n_bins=13, n_jobs=3, ccp_alpha=0.0, random_state=None):
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None
        self.shapelets_values = None
        self.ccp_alpha= ccp_alpha
        self.P = P
        self.n_trees = n_trees
        self.use_class_weights=use_class_weights
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
        
    def fit(self,X,y):
        X = check_array_3D(X)
        #L, dils, X_split, y_split, kernel_id = self._generate_inputs(X,y)
        m = MiniRocket().fit(X)
        ft, L = m.transform(X, return_locs=True)
        self._log("Performing kernel selection with {} kernels".format(L.shape[1]))
        if self.use_class_weights:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        class_weight='balanced',
                                        ccp_alpha=self.ccp_alpha,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state)
        else:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        ccp_alpha=self.ccp_alpha,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state)
        rf.fit(ft, y)
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=np.uint16)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n += 84*num_features_per_dilation[i]
        
        n_nodes = np.asarray([rf.estimators_[i_dt].tree_.node_count 
                              for i_dt in range(self.n_trees)])
        n_leaves = np.asarray([rf.estimators_[i_dt].tree_.n_leaves 
                               for i_dt in range(self.n_trees)])
        n_splits = (n_nodes - n_leaves).sum()
        
        kernel_id = np.zeros((n_splits),dtype=np.int32)
        X_split = np.zeros((n_splits,X.shape[0]),dtype=np.bool_)
        y_split = np.zeros((n_splits,X.shape[0]),dtype=np.int8) - 1
        
        i_split_nodes = np.zeros(self.n_trees+1,dtype=np.int32)
        i_split_nodes[1:] += (n_nodes - n_leaves).cumsum()
        
        for i_dt in range(self.n_trees):
            tree = rf.estimators_[i_dt].tree_
            node_indicator = rf.estimators_[i_dt].decision_path(ft)
            nodes_id = np.where(tree.feature != _tree.TREE_UNDEFINED)[0]
            kernel_id[i_split_nodes[i_dt]:i_split_nodes[i_dt+1]] += tree.feature[nodes_id]
            for i in range(nodes_id.shape[0]):
                x_index_node = node_indicator[:,nodes_id[i]].nonzero()[0]
                X_split[i_split_nodes[i_dt]+i, x_index_node] = True
                y_split[i_split_nodes[i_dt]+i, x_index_node] += (ft[
                    x_index_node, kernel_id[i_split_nodes[i_dt]+i]
                    ] <= tree.threshold[nodes_id[i]]) + 1
        
        arr = np.concatenate((X_split,y_split,kernel_id.reshape(-1,1)),axis=1)
        v_unique, i_unique = np.unique(arr,axis=0,return_index=True)
        X_split = X_split[i_unique]
        y_split = y_split[i_unique]
        kernel_id = kernel_id[i_unique]
        
        shp, d = _fit(X, X_split, y_split, kernel_id, L, dils, self.P)
        self.n_shapelets = 0
        self.shapelets = {}
        for dil in np.unique(d):
            candidates = shp[d==dil]
            candidates = (candidates - candidates.mean(axis=-1, keepdims=True)) / (
                candidates.std(axis=-1, keepdims=True) + 1e-8)
            if not np.all(candidates.reshape(-1, 1) == candidates.reshape(-1, 1)[0]):
                kbd = KBinsDiscretizer(n_bins=self.n_bins, strategy='uniform', dtype=np.float32).fit(
                    candidates.reshape(-1, 1))
                candidates = np.unique(kbd.inverse_transform(
                    kbd.transform(candidates.reshape(-1, 1))).reshape(-1, 9), axis=0)
            else:
                candidates = np.unique(candidates, axis=0)
                
            self.n_shapelets += candidates.shape[0]
            self.shapelets.update({dil: candidates})
        return self
    
    def transform(self, X, store=False, return_inverse=False):
        X = check_array_3D(X)
        distances = np.zeros((X.shape[0], self.n_shapelets),dtype=np.float32)
        prev = 0
        if store:
            self.shp = []
            self.dil = []
        for i, dil in enumerate(self.shapelets.keys()):
            self._log("Transforming for dilation {} ({}/{}) with {} shapelets".format(
                dil, i, len(self.shapelets), len(self.shapelets[dil])))
            dilation = dil
            X_strides = _generate_strides_2d(X[:,self.id_ft,:], 9, dilation)
            X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
                X_strides.std(axis=-1, keepdims=True) + 1e-8)
            if store:
                self.shp.extend(self.shapelets[dil])
                self.dil.extend([dil]*self.shapelets[dil].shape[0])
            d = np.asarray([cdist(X_strides[j], self.shapelets[dil],
                                  metric='sqeuclidean').min(axis=0)
                            for j in range(X.shape[0])])
            distances[:, prev:prev+d.shape[1]] = d
            prev += d.shape[1]
        if return_inverse:
            return 1/(distances+1e-8)
        else:
            return distances
    
    def _generate_inputs(self, X, y):
        self._log("Performing MiniRocket Transform")
        m = MiniRocket().fit(X)
        ft, L = m.transform(X, return_locs=True)
        self._log("Performing kernel selection with {} kernels".format(L.shape[1]))
        if self.use_class_weights:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        class_weight='balanced',
                                        ccp_alpha=self.ccp_alpha,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state)
        else:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        ccp_alpha=self.ccp_alpha,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state)
        rf.fit(ft, y)
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=int)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n += 84*num_features_per_dilation[i]
        
        n_nodes = np.asarray([rf.estimators_[i_dt].tree_.node_count 
                              for i_dt in range(self.n_trees)])
        n_leaves = np.asarray([rf.estimators_[i_dt].tree_.n_leaves 
                               for i_dt in range(self.n_trees)])
        n_splits = (n_nodes - n_leaves).sum()
        
        kernel_id = np.zeros((n_splits),dtype=np.uint16)
        X_split = np.zeros((n_splits,X.shape[0]),dtype=np.bool_)
        y_split = np.zeros((n_splits,X.shape[0]),dtype=np.int8) - 1
        
        i_split_nodes = np.zeros(self.n_trees+1,dtype=np.int32)
        i_split_nodes[1:] += (n_nodes - n_leaves).cumsum()
        
        for i_dt in range(self.n_trees):
            tree = rf.estimators_[i_dt].tree_
            node_indicator = rf.estimators_[i_dt].decision_path(ft)
            nodes_id = np.where(tree.feature != _tree.TREE_UNDEFINED)[0]
            kernel_id[i_split_nodes[i_dt]:i_split_nodes[i_dt+1]] += tree.feature[nodes_id]
            for i in range(nodes_id.shape[0]):
                x_index_node = node_indicator[:,nodes_id[i]].nonzero()[0]
                X_split[i_split_nodes[i_dt]+i, x_index_node] = True
                y_split[i_split_nodes[i_dt]+i, x_index_node] += (ft[
                    x_index_node, kernel_id[i_split_nodes[i_dt]+i]
                    ] <= tree.threshold[nodes_id[i]]) + 1

        return L, dils, X_split, y_split, kernel_id
   
#TODO : When numba support sparse array, make sparse here !
@njit(cache=True, fastmath=True, parallel=True)
def _fit(X, X_split, y_split, kernel_id, L, dils, P):
    x_mask = np.zeros((X_split.shape[0],2,X.shape[0]*X.shape[2]),dtype=np.bool_)
    for i_split in prange(X_split.shape[0]):
        x_indexes = np.where(X_split[i_split])[0]
        k_id = kernel_id[i_split]
        y_splt = y_split[i_split][x_indexes]
        dil = dils[k_id]
        classes = np.unique(y_splt)
        n_classes = classes.shape[0]
        Lp = _generate_strides_2d(L[x_indexes, k_id, :], 9, dil).sum(axis=-1)
        c_w =  X[x_indexes].shape[0] / (n_classes * np.bincount(y_splt))
        
        LC = np.zeros((n_classes, Lp.shape[1]))
        for i_class in classes:
            LC[i_class] = c_w[i_class] * Lp[y_splt==i_class].sum(axis=0)
        
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
                        x_mask[i_split, i_class, x_indexes[x_index]*X.shape[2] + id_max_region] += 1
    
    n_candidates = np.sum(x_mask)
    candidates = np.zeros((n_candidates,9),dtype=np.float32)
    candidates_dil = np.zeros((n_candidates),dtype=np.uint16)
    per_split_id = np.zeros((X_split.shape[0]+1),dtype=np.int32)
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
            indexes = np.where(x_mask[i_split,i_class])[0]
            
            x_indexes = indexes//X.shape[2]
            l_indexes = indexes%X.shape[2]
            for i_candidate in prange(x_indexes.shape[0]):
                for j in prange(9):
                    
                    candidates[per_split_id[i_split]+(i_class*(x_mask[i_split,0].sum()))+i_candidate 
                               ,j] = X[x_indexes[i_candidate],0, l_indexes[i_candidate]+j*dil]
    
    return candidates, candidates_dil

@njit(cache=True)
def _get_regions(indexes):
    regions = np.zeros((indexes.shape[0]*2),dtype=np.int32)-1
    p = 0
    for i in prange(indexes.shape[0]-1):
        regions[p] = indexes[i]
        if indexes[i] == indexes[i+1]-1:
            p+=1
        else:
            p+=2
    regions[p] = indexes[-1]
    idx = np.where(regions!=-1)[0]
    return regions[idx], np.concatenate((np.array([0],dtype=np.int32),np.where(np.diff(idx)!=1)[0]+1,np.array([indexes.shape[0]],dtype=np.int32)))

@njit(cache=True)
def _generate_strides_2d(X, window_size, window_step):
    n_samples, n_timestamps = X.shape
    
    shape_new = (n_samples,
                 n_timestamps - (window_size-1)*window_step,
                 window_size // 1)
    s0, s1 = X.strides
    strides_new = (s0, s1, window_step *s1)
    return as_strided(X, shape=shape_new, strides=strides_new)
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:52:44 2021

@author: Antoine
"""
import numpy as np

from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, generate_strides_1D
from CST.utils.checks_utils import check_array_3D
#from CST.base_transformers.minirocket import MiniRocket
from itertools import combinations

from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle


class MiniConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_ft=0, verbose=0):
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None    
        self.shapelets_values = None
                 
    def _log(self, message):
        if self.verbose > 0:
            print(message)
    
    #TODO : Add a value mapping to handle case where difference is made by value and not location
    #TODO : review parameter names
    def fit(self, X, y, n_bins=8, percentile_select=90, n_shp_per_comb=4):
        X = check_array_3D(X)
        n_classes = np.unique(y).shape[0]
        n_comb = len(list(combinations(range(n_classes),2)))
        
        locs, dils, biases = self._init_kernels(X, y)
        biases = np.array([b>=0 for b in biases]).astype(int)
        groups_id, unique_groups = self._get_kernel_groups(dils, biases)
        
        shapelets_values = np.zeros((locs.shape[1],n_comb * n_shp_per_comb,9))
 
        for i_grp in unique_groups.keys():
            values_grp = []
            for c1,c2 in combinations(range(n_classes),2):
                id_c1 = np.array_split(shuffle(np.where(y==c1)[0]),n_shp_per_comb)
                id_c2 = np.array_split(shuffle(np.where(y==c2)[0]),n_shp_per_comb)
                for i_iter in range(n_shp_per_comb):
                    loc_c1 = np.sum(locs[id_c1[i_iter]],axis=0)
                    loc_c2 = np.sum(locs[id_c2[i_iter]],axis=0)
                    loc_c1 = (loc_c1 - loc_c1.min()) / (loc_c1.max() - loc_c1.min())
                    loc_c2 = (loc_c2 - loc_c2.min()) / (loc_c2.max() - loc_c2.min())
                    
                    id_1 = self._select_id_loc(loc_c1,loc_c2,1,
                                               percentile_select=percentile_select)
                    id_2 = self._select_id_loc(loc_c2,loc_c1,1,
                                               percentile_select=percentile_select)
                    
                    for i in np.concatenate([id_1,id_2]):
                        #Look at loc_c1/c2 values to build candidate, ie take all possible i +/-dil with maximum sum value
                        values_grp.append()
            
            values_grp = np.concatenate(values_grp,axis=0)
            if values_grp.shape[0] > 0:
                n_candidates = values_grp.shape[0]
                kbd = KBinsDiscretizer(n_bins=n_bins, strategy='uniform').fit(values_grp.reshape(-1,1))
                values_grp = np.unique(kbd.inverse_transform(kbd.transform(values_grp.reshape(-1,1))).reshape(-1,9),axis=0)
                self._log("Discretization filtered {}/{} candidates".format(n_candidates-values_grp.shape[0],n_candidates))
                
                values.update({i_grp : values_grp})
                self._log("Extracted {} candidates from {} kernels".format(
                    values_grp.shape[0],grp_locs.shape[0]))    
                
        self.shapelets_params = {i_grp : unique_groups[i_grp] for i_grp in values.keys()}
        self.shapelets_values = values
        return self
        
    def transform(self, X):
        self._check_is_fitted()
        X = check_array_3D(X)
        distances = []
        for i, i_grp in enumerate(self.shapelets_params.keys()):
            self._log("Transforming for grp {} ({}/{}) with {} shapelets".format(self.shapelets_params[i_grp],
                                                                                 i, len(self.shapelets_params),
                                                                                 self.shapelets_values[i_grp].shape[0]))
            dilation, _ = self.shapelets_params[i_grp]
            X_strides = self._get_X_strides(X, 9 ,dilation, 0)
            distances.append(compute_distances(X_strides, self.shapelets_values[i_grp]))
        return np.concatenate(distances, axis=1)
    
    
    def _select_id_loc(self, loc_c1, loc_c2, n_shapelet_per_combination,
                       percentile_select = 90):
        diff = loc_c1 - loc_c2
        id_conv = np.where(diff>=np.percentile(diff,percentile_select))[0]
        n = n_shapelet_per_combination if n_shapelet_per_combination < id_conv.shape[0] else id_conv.shape[0]
        return np.random.choice(id_conv, n, replace=False)
    
    def _get_kernel_groups(self, kernels_dilations, kernels_bias):
        groups_params = np.array([[kernels_dilations[i], 
                                   kernels_bias[i]]
                                  for i in range(kernels_dilations.shape[0])])
                                 
        groups_id = np.zeros(kernels_dilations.shape[0])
        unique_groups = np.unique(groups_params, axis=0)
        unique_groups = {i: unique_groups[i] for i in range(unique_groups.shape[0])}
        
        for i in unique_groups.keys():
            groups_id[np.where((groups_params == unique_groups[i]).all(axis=1))[0]] = i
        return groups_id, unique_groups
    
    def _get_X_strides(self, X, length, dilation, padding):
        n_samples, _, n_timestamps = X.shape
        if padding > 0:
            X_pad = np.zeros((n_samples, n_timestamps+2*padding))
            X_pad[:,padding:-padding] = X[:,self.id_ft,:]
        else:
            X_pad = X[:,self.id_ft,:]
        X_strides = generate_strides_2D(X_pad,length,dilation)
        X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
                        X_strides.std(axis=-1, keepdims=True) + 1e-8)
        return X_strides
    
    def _get_idx_strides(self, X, length, dilation, padding):
        _, _, n_timestamps = X.shape
        n_timestamps += 2*padding
        x_idx = generate_strides_1D(np.array(range(n_timestamps)),length,dilation) - padding
        x_idx[np.where((x_idx >= X.shape[2]) | (x_idx < 0))] = -1
        return x_idx 
    
    def _init_kernels(self, X, y):
        print("RKT ...")
        m = MiniRocket().fit(X)
        ft, locs = m.transform(X, return_locs=True)
        print("Selection ...")
        
        ft_selector = SelectFromModel(RandomForestClassifier(max_features=0.75,
                                                             max_samples=0.75,
                                                             ccp_alpha=0.015,
                                                             n_jobs=-1)).fit(ft,y)
        print("Done ...")
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=int)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n+=84*num_features_per_dilation[i]
            
        i_kernels = np.where(ft_selector.get_support())[0]

        return locs[:, i_kernels], dils[i_kernels], biases[i_kernels] 
        #MINIROCKET, then select by model with RF, extract relevant ft location
    
    def _check_is_fitted(self):
        if any(self.__dict__[attribute] is None for attribute in ['shapelets_values',
                                                                  'shapelets_params']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")
            
        
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import f1_score

X_train, X_test, y_train, y_test,_ = load_sktime_dataset_split('ACSF1')
m = MiniConvolutionalShapeletTransformer(verbose=1).fit(X_train, y_train)
X_rkt_train = m.transform(X_train)
X_rkt_test = m.transform(X_test)
rf = RandomForestClassifier(n_estimators=400).fit(X_rkt_train, y_train)
preds = rf.predict(X_rkt_test)
print(f1_score(y_test, preds))



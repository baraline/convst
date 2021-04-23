# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:52:44 2021

@author: Antoine
"""
import numpy as np

from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, generate_strides_1D
from CST.utils.checks_utils import check_array_3D
from CST.base_transformers.minirocket import MiniRocket
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle

#TODO : Add a value mapping to handle case where difference is made by raw conv value density and not location
#TODO : try to compare one class vs all rather than one vs one 
class MiniConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_ft=0, verbose=0):
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None    
        self.shapelets_values = None
                 
    def _log(self, message):
        if self.verbose > 0:
            print(message)
            
    def _log2(self, message):
        if self.verbose > 1:
            print(message)
    
    def fit(self, X, y, n_bins=9, p=90, n_splits=3,
            p_samples_to_shp_vals=0.1, n_locs_per_split=2):
        X = check_array_3D(X)
        #locs = (n_samples, n_kernels, n_timestamps)
        locs, dils, biases = self._init_kernels(X, y)
        groups_id, unique_groups = self._get_kernel_groups(dils, biases)
        self._log("Begining extraction with {} kernel groups".format(len(unique_groups)))
        classes = set(np.unique(y))
        n_timestamps = X.shape[2]
        n_classes = np.unique(y).shape[0]
                
        n_shapelets = 0
        values = {}
        for i_grp in unique_groups.keys():
            dilation = int(unique_groups[i_grp][0])
            values_grp = []
            locs_grps = np.zeros((n_classes, n_splits, n_timestamps),dtype=np.float32)
            id_splits = []
            #Generate convolution input localisation for each class and split
            #TODO: balanced vals by number of sample per class
            for i_class in np.unique(y):
                id_splits.append(np.array_split(shuffle(np.where(y==i_class)[0]),n_splits))
                for i_split in range(n_splits):
                    vals = np.sum(np.sum(locs[id_splits[i_class][i_split], np.where(groups_id==i_grp)[0][:,None], :],axis=0),axis=0)
                    locs_grps[i_class, i_split] = vals
                    #TODO : Test without normalization
                    #if vals.max() != vals.min():
                    #    np.divide(vals - vals.min(), vals.max()-vals.min(), out=locs_grps[i_class, i_split])
            
            for c in classes:
                for i_split in range(n_splits):
                    diff_other_class = np.asarray([locs_grps[j, i_split, :] 
                                        for j in classes - {c} 
                                        if locs_grps[j, i_split, :].sum() != 0])
                    diff_other_class = diff_other_class.mean(axis=0)
                    diff = locs_grps[c, i_split, :] - diff_other_class
                    stride_sums = generate_strides_1D(diff, 9, dilation).sum(axis=1)
                    id_strides = np.where(stride_sums >= np.percentile(stride_sums,p))[0]
                    value_locs = np.random.choice(id_strides, n_locs_per_split if n_locs_per_split <= id_strides.shape[0] else id_strides.shape[0], replace=False)
                    # For each random candidate location, extract 
                    for i in value_locs:
                        
                        id_x = np.random.choice(id_splits[c][i_split],
                                                int(np.ceil(id_splits[c][i_split].shape[0]*p_samples_to_shp_vals)),
                                                replace=False)
                        for ix in id_x:
                            values_grp.append(X[ix,0,np.array([i+j*dilation for j in range(9)])])
            values_grp = np.asarray(values_grp)
            self._log2("Got {} candidates for grp {}".format(values_grp.shape[0], i_grp))               
            if values_grp.shape[0] > 0 and not np.all(values_grp == values_grp[0][0]):
                values_grp = (values_grp - values_grp.mean(axis=-1, keepdims=True)) / (
                    values_grp.std(axis=-1, keepdims=True) + 1e-8)
                kbd = KBinsDiscretizer(n_bins=n_bins, strategy='uniform').fit(values_grp.reshape(-1,1))
                values_grp = np.unique(kbd.inverse_transform(kbd.transform(values_grp.reshape(-1,1))).reshape(-1,9),axis=0)
                n_shapelets += values_grp.shape[0]
                values.update({i_grp : values_grp})
                self._log("Extracted {} shapelets for grp {}/{}".format(values_grp.shape[0], i_grp, len(unique_groups.keys())))
        
        
        self.shapelets_params = {i_grp : unique_groups[i_grp] for i_grp in values.keys()}
        self.shapelets_values = values
        self._log("Extracted a total of {} shapelets".format(n_shapelets))
        self.n_shapelets = n_shapelets
        return self
    
    def transform(self, X):
        self._check_is_fitted()
        X = check_array_3D(X)
        distances = np.zeros((X.shape[0], self.n_shapelets))
        prev = 0
        for i, i_grp in enumerate(self.shapelets_params.keys()):
            self._log("Transforming for grp {} ({}/{}) with {} shapelets".format(self.shapelets_params[i_grp],
                                                                                 i, len(self.shapelets_params),
                                                                                 self.shapelets_values[i_grp].shape[0]))
            dilation, _ = self.shapelets_params[i_grp]
            X_strides = self._get_X_strides(X, 9 ,dilation, 0)
            d = compute_distances(X_strides, self.shapelets_values[i_grp])            
            distances[:, prev:prev+d.shape[1]] = d
            prev += d.shape[1]
        return distances
        
    def _get_kernel_groups(self, kernels_dilations, kernels_bias):
        kernels_bias = np.array([b>=0 for b in kernels_bias]).astype(int)
        groups_params = np.array([[kernels_dilations[i], 
                                   kernels_bias[i]]
                                  for i in range(kernels_dilations.shape[0])],dtype=np.int32)
        
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
        
    def _init_kernels(self, X, y):
        self._log("Performing MiniRocket Transform")
        m = MiniRocket().fit(X)
        ft, locs = m.transform(X, return_locs=True)
        self._log("Performing kernel selection with {} kernels".format(locs.shape[1]))
        
        ft_selector = SelectFromModel(RandomForestClassifier(max_features=0.85,
                                                             max_samples=0.85,
                                                             ccp_alpha=0.02,
                                                             n_jobs=None)).fit(ft,y)
        
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=int)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n+=84*num_features_per_dilation[i]
            
        i_kernels = np.where(ft_selector.get_support())[0]
        self._log("Finished kernel selection with {} kernels".format(i_kernels.shape[0]))
        return locs[:, i_kernels], dils[i_kernels], biases[i_kernels] 
    
    def _check_is_fitted(self):
        if any(self.__dict__[attribute] is None for attribute in ['shapelets_values',
                                                                  'shapelets_params']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")
        
                
    
    
        
        
    

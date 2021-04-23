# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 08:52:54 2021

@author: A694772
"""
import numpy as np

from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, generate_strides_1D
from CST.factories.kernel_factories import Rocket_factory
from CST.utils.checks_utils import check_array_3D
from CST.base_transformers.rocket import ROCKET

from itertools import combinations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

#TODO : Update this to adopt the algorithm used in mini_CST
class ConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_ft=0, verbose=0):
        self.rkt_object = None
        self.k_factory = None
        self.ft_imps = None
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None    
        self.shapelets_values = None
    
    def _init_rkt(self, X, y):
        rkt_object = ROCKET(kernel_sizes=[9]).fit(X,y)
        X_rkt = rkt_object.transform(X)
        rf = RandomForestClassifier(n_estimators=400, max_features=0.5, max_samples=0.75, ccp_alpha=0.015).fit(X_rkt, y)
        self.rkt_object = rkt_object
        self.ft_imps = rf.feature_importances_
        self.k_factory = Rocket_factory(rkt_object)
        
    #TODO : Add a value mapping to handle case where difference is made by value and not location
    #TODO : review parameter names
    def fit(self, X, y, n_shapelet_per_combination=2, n_iter_per_comb=5, 
            n_bins=8, percentile_select=90):
        
        X = check_array_3D(X)
        self._init_rkt(X, y)
        self._check_is_init()
        
        kernels, kernels_dilations, kernels_length, kernels_bias, kernels_padding = self._get_kernels()
        self._log("Selected {} features out of {}".format(kernels.shape[0],
                                                          self.ft_imps.shape[0]))
        
        
        groups_id, unique_groups = self._get_kernel_groups(kernels_dilations,
                                                           kernels_length,
                                                           kernels_bias, 
                                                           kernels_padding)
                                                           
        n_classes = np.unique(y).shape[0]
        max_dilation = np.max(np.array(list(unique_groups.values()))[:,0])
        
        #TODO : clean the debugging mess
        self._log("Max dilation : {}".format(max_dilation))
        if max_dilation > 1:
            dil_to_consider = self.primesfrom2to(max_dilation)
            print(dil_to_consider)
            to_pop = []
            for i_grp in unique_groups.keys():
                grp_dilation = unique_groups[i_grp][0]
                if  grp_dilation != 1 and grp_dilation not in dil_to_consider:
                    to_pop.append(i_grp)
            self._log("Filtered {}/{} kernel groups based on dilation".format(len(to_pop),len(unique_groups)))
            for key in to_pop:
                unique_groups.pop(key, None)
        values = {}
        for i_grp in unique_groups.keys():
            dilation, length, bias, padding = unique_groups[i_grp]
            
            self._log('-------------------')
            self._log("Extracting Shapelets for kernels with {}".format(str(unique_groups[i_grp])))
            grp_kernels = kernels[np.where(groups_id == i_grp)[0]]
            X_conv_indexes = self._get_idx_strides(X, length, dilation, padding)
            X_locs = self._get_X_locs(X, grp_kernels)
            X_strides = self._get_X_strides(X, length ,dilation, padding)
            self._log("Strides shape {}".format(X_strides.shape))                        
            
            values_grp = []
            for c1,c2 in combinations(range(n_classes),2):
                #Create random partition of each class according to n_iter_per_comb param
                id_c1 = np.array_split(shuffle(np.where(y==c1)[0]),n_iter_per_comb)
                id_c2 = np.array_split(shuffle(np.where(y==c2)[0]),n_iter_per_comb)
                for i_iter in range(n_iter_per_comb):
                    loc_c1 = np.sum(X_locs[id_c1[i_iter]],axis=0)
                    loc_c1 = (loc_c1 - loc_c1.min()) / (
                        loc_c1.max() - loc_c1.min())
                    loc_c2 = np.sum(X_locs[id_c2[i_iter]],axis=0)
                    loc_c2 = (loc_c2 - loc_c2.min()) / (
                        loc_c2.max() - loc_c2.min())

                    
                    id_x = self._select_id_loc(loc_c1,loc_c2,n_shapelet_per_combination, 
                                               percentile_select=percentile_select)
                    for i in id_x:
                        id_to_get = np.where(X_conv_indexes == i)[0]
                        id_to_get = np.random.choice(id_to_get, int(np.ceil(id_to_get.shape[0]*0.2)))
                        id_x_to_get = np.random.choice(id_c1[i_iter], int(np.ceil(id_c1[i_iter].shape[0]*0.2)))
                        values_grp.append((X_strides[id_x_to_get[:,None], id_to_get]).reshape(-1, length))
                        
                    id_x = self._select_id_loc(loc_c2,loc_c1,n_shapelet_per_combination,
                                               percentile_select=percentile_select)
                    for i in id_x:
                        id_to_get = np.where(X_conv_indexes == i)[0]
                        id_to_get = np.random.choice(id_to_get, int(np.ceil(id_to_get.shape[0]*0.2)))
                        id_x_to_get = np.random.choice(id_c2[i_iter], int(np.ceil(id_c2[i_iter].shape[0]*0.2)))
                        values_grp.append((X_strides[id_x_to_get[:,None], id_to_get]).reshape(-1, length))
            
            values_grp = np.concatenate(values_grp,axis=0)
            if values_grp.shape[0] > 0:
                values.update({i_grp : values_grp})
            self._log("Extracted {} candidates from {} kernels".format(
                values_grp.shape[0],grp_kernels.shape[0]))     
        self._log('-------------------')
        
        k_params = np.array(list(unique_groups.values()))
        print(k_params.shape)
        bin_params = np.unique(np.array([[k_params[i,0], k_params[i,1]] 
                                         for i in range(k_params.shape[0])]),axis=0)
        for bin_p in bin_params:
            id_bins = np.array([i_grp for i_grp, grp_params in unique_groups.items() if np.array_equal(grp_params[[0,1]],bin_p)])
            
            values_grp = np.array([values[id_grp] for id_grp in id_bins],dtype='object')
            all_values = np.concatenate(values_grp,axis=0)
            n_candidates = all_values.shape[0]
            
            kbd = KBinsDiscretizer(n_bins=n_bins, strategy='uniform').fit(all_values.reshape(-1,1))
            
            all_values = kbd.inverse_transform(kbd.transform(all_values.reshape(-1,1))).reshape(-1,bin_p[1])
            
            unique_vals = np.unique(all_values,axis=0)
            values_grp = np.array([kbd.inverse_transform(kbd.transform(values_grp[id_grp].reshape(-1,1))).reshape(-1,bin_p[1]) for id_grp in range(id_bins.shape[0])],dtype='object')

            self._log("Discretization filtered {}/{} candidates for groups {}".format(n_candidates-unique_vals.shape[0],n_candidates,bin_p))
            
            affect_to_id = np.zeros(unique_vals.shape[0]) - 1
            
            for i_val in range(affect_to_id.shape[0]):
                for id_grp in range(id_bins.shape[0]):
                    if affect_to_id[i_val] == -1 and any(np.equal(values_grp[id_grp],unique_vals[i_val]).all(axis=1)):
                        affect_to_id[i_val] = id_grp
            
            for id_grp in range(id_bins.shape[0]):
                id_affect = np.where(affect_to_id==id_grp)[0]
                values[id_bins[id_grp]] = unique_vals[id_affect]
                if id_affect.shape[0] == 0:
                    print("Dropped {}".format(id_bins[id_grp]))
                    values.pop(id_bins[id_grp], None)
        
        
        to_keep = list(values.keys())
        g_keys = list(unique_groups.keys())
        for key in g_keys:
            if key not in to_keep:
                unique_groups.pop(key, None)
        self.shapelets_params = unique_groups
        self.shapelets_values = values
        
        #TODO : Only a fraction of the selected candidates are used in while learning a model could a selection method be used ?
        
        return self    
    
    def primesfrom2to(self, n):
        """ Input n>=6, Returns a array of primes, 2 <= p < n """
        if n <= 1:
            return []
        elif n == 2:
            return [2]
        elif n == 3:
            return [2,3]
        sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
        sieve[0] = False
        for i in range(int(n**0.5)//3+1):
            if sieve[i]:
                k=3*i+1|1
                sieve[      ((k*k)//3)      ::2*k] = False
                sieve[(k*k+4*k-2*k*(i&1))//3::2*k] = False
        return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]
    
    def transform(self, X):
        self._check_is_fitted()
        X = check_array_3D(X)
        distances = []
        #TODO : Distance computation can be done ignoring the bias
        #TODO : Could vectorization speedups / GPU speedups be used for distances computations
        for i, i_grp in enumerate(self.shapelets_params.keys()):
            self._log("Transforming for grp {} ({}/{}) with {} shapelets".format(self.shapelets_params[i_grp],
                                                                                 i, len(self.shapelets_params),
                                                                                 self.shapelets_values[i_grp].shape[0]))
            dilation, length, _, padding = self.shapelets_params[i_grp]
            X_strides = self._get_X_strides(X, length ,dilation, padding)
            distances.append(compute_distances(X_strides, self.shapelets_values[i_grp]))
        return np.concatenate(distances, axis=1)
             
    def _log(self, message):
        if self.verbose > 0:
            print(message)
    
    def _select_id_loc(self, loc_c1, loc_c2, n_shapelet_per_combination,
                       percentile_select = 90):
        diff = loc_c1 - loc_c2
        id_conv = np.where(diff>=np.percentile(diff,percentile_select))[0]
        n = n_shapelet_per_combination if n_shapelet_per_combination < id_conv.shape[0] else id_conv.shape[0]
        return np.random.choice(id_conv, n, replace=False)
        
    
    def _get_X_locs(self, X, kernels):
        X_locs = np.zeros((X.shape[0], X.shape[2]))
        for i_k in range(kernels.shape[0]):
            locs = kernels[i_k].get_locs(X)
            for i_x in range(X.shape[0]):
                X_locs[i_x, locs[i_x].astype(int)] += 1
        return X_locs
            
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
    
    
    def _get_kernels(self):
        kernel_ids = np.where(self.ft_imps > 0.000025)[0]
        kernels = np.asarray([self.k_factory.create_feature_kernel(k_id) 
                              for k_id in kernel_ids])
        
        kernels_dilations = np.asarray([kernels[k_id].dilation for k_id in range(kernels.shape[0])])
        kernels_length = np.asarray([kernels[k_id].length for k_id in range(kernels.shape[0])])
        kernels_bias = np.asarray([kernels[k_id].bias >= 0 for k_id in range(kernels.shape[0])]).astype(int)
        kernels_padding = np.asarray([kernels[k_id].padding > 0 for k_id in range(kernels.shape[0])]).astype(int)
        
        return kernels, kernels_dilations, kernels_length, kernels_bias, kernels_padding
    
    #TODO Need to test performance when removing some of the grouping conditions (bias/padding mostly)
    def _get_kernel_groups(self, kernels_dilations, kernels_length, kernels_bias, kernels_padding):
        groups_params = np.array([[kernels_dilations[i], 
                                   kernels_length[i], 
                                   kernels_bias[i],
                                   kernels_padding[i]]
                                  for i in range(kernels_dilations.shape[0])])
                                 
        groups_id = np.zeros(kernels_dilations.shape[0])
        unique_groups = np.unique(groups_params, axis=0)
        unique_groups = {i: unique_groups[i] for i in range(unique_groups.shape[0])}
        
        for i in unique_groups.keys():
            groups_id[np.where((groups_params == unique_groups[i]).all(axis=1))[0]] = i
        return groups_id, unique_groups
    
    def _check_is_init(self):
        if any(self.__dict__[attribute] is None for attribute in ['rkt_object',
                                                                  'ft_imps',
                                                                  'id_ft']):
            raise AttributeError("CST attributes not initialised correctly, "
                                 "at least one attribute was set to None")
    
    def _check_is_fitted(self):
        if any(self.__dict__[attribute] is None for attribute in ['shapelets_values',
                                                                  'shapelets_params']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")                
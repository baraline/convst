# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 08:52:54 2021

@author: A694772
"""
import numpy as np

from CST.utils.shapelets_utils import compute_distances, generate_strides_2D, generate_strides_1D
from CST.factories.kernel_factories import Rocket_factory

from CST.utils.checks_utils import check_array_3D

from itertools import combinations

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import shuffle

class ConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rkt_object, ft_imps, id_ft=0, verbose=0):
        self.rkt_object = rkt_object
        self.k_factory = Rocket_factory(rkt_object)
        self.ft_imps = ft_imps
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None    
        self.shapelets_values = None
    
    #TODO : Add a value mapping to handle case where difference is made by value and not location
    def fit(self, X, y, n_shapelet_per_combination=2, n_iter_per_comb=4, 
            n_bins=10, percentile_select=90):
        self._check_is_init()
        X = check_array_3D(X)
        
        kernels, kernels_dilations, kernels_length, kernels_bias, kernels_padding = self._get_kernels()
        self._log("Selected {} features out of {}".format(kernels.shape[0],
                                                          self.ft_imps.shape[0]))
        
        
        groups_id, unique_groups = self._get_kernel_groups(kernels_dilations,
                                                           kernels_length,
                                                           kernels_bias, 
                                                           kernels_padding)
        n_classes = np.unique(y).shape[0]
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
                        values_grp.append((X_strides[id_c1[i_iter][:,None], np.where(X_conv_indexes == i)[0]]).reshape(-1, length))
                        
                    id_x = self._select_id_loc(loc_c2,loc_c1,n_shapelet_per_combination,
                                               percentile_select=percentile_select)
                    for i in id_x:
                        values_grp.append((X_strides[id_c2[i_iter][:,None], np.where(X_conv_indexes == i)[0]]).reshape(-1, length))    
                    
            values_grp = np.concatenate(values_grp,axis=0)
            if values_grp.shape[0] > 0:
                values.update({i_grp : values_grp})
            self._log("Extracted {} candidates from {} kernels".format(
                values_grp.shape[0],grp_kernels.shape[0]))     
        self._log('-------------------')
        bin_params = np.unique(np.array([[kernels_dilations[i], kernels_length[i]] 
                                         for i in range(kernels_dilations.shape[0])]),axis=0)
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
    
    def transform(self, X):
        self._check_is_fitted()
        X = check_array_3D(X)
        distances = []
        #TODO : Distance computation can be done ignoring the bias
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
        kernel_ids = np.where(self.ft_imps > 0.00001)[0]
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
            
"""
class ConvolutionalShapeletTransformClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_kernels=10000, kernel_sizes=(7, 9, 11),
                 random_state=None, id_ft=0, verbose=0):
        self.n_kernels = n_kernels
        self.id_ft = id_ft
        self.verbose = verbose
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state
        self.shapelets = []
    
    def _log(self, message):
        if self.verbose > 0:
            print(message)
    
    def _log2(self, message):
        if self.verbose > 1:
            print(message)
    
    def fit(self, X, y, n_bins=12, n_max_kernel=None, bin_all=False, use_random_selection=None):
        self._check_is_init()

        X = self._check_array(X)
        self._log("Creating ROCKET features ...")    
        rkt = ROCKET(n_kernels=self.n_kernels, kernel_sizes=self.kernel_sizes,
                     id_ft=self.id_ft, random_state=self.random_state)
        X_rkt = rkt.fit_transform(X)
        rfc = RandomForestClassifier(n_estimators=400, ccp_alpha=0.015, max_samples=0.5).fit(X_rkt, y)
        id_ft = np.where(rfc.feature_importances_>0)[0]
        if n_max_kernel is None:
            id_ft = np.where(rfc.feature_importances_ >= rfc.feature_importances_[id_ft].mean())[0] 
        else:
            id_ft = id_ft[np.argsort(rfc.feature_importances_[id_ft])[::-1][0:n_max_kernel]]
        self._log("Extracting feature kernels ...")    
        self.rocket_factory = Rocket_factory(rkt)
        kernels = np.array([self.rocket_factory.create_feature_kernel(i) for i in id_ft])
        self._log("Extracted {} feature kernels".format(kernels.shape[0]))
        
        params_k = np.asarray([[k.padding,k.dilation] for k in kernels])
        
        ##########################################
        #                                        #
        #  FROM HERE ONLY SUPPORT 3D univariate  #
        #                                        #
        ##########################################
        
        all_values = []
        all_distances = []
        #TODO : add another loop per kernel lenght
        all_params_id = [0]
        all_params = []
        for k in np.unique(params_k,axis=0):
            idx = np.where((params_k==k).all(axis=1))[0]

            self._log2("{} kernels for params {}".format(idx.size,k))
            self._log2("------------")
            values, X_strides = self._input_value_matrix_from_ft_kernelv2(X,kernels[idx])
            if values.size > 0:
                values = (values - values.mean(axis=-1, keepdims=True)) / (
                        values.std(axis=-1, keepdims=True) + 1e-8)
                X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
                        X_strides.std(axis=-1, keepdims=True) + 1e-8)
                
                kbd = KBinsDiscretizer(n_bins=n_bins)
                self._log2("Extracted {} candidates from {} kernels".format(
                    values.shape,idx.size))     
                if bin_all:
                    kbd.fit(values.reshape(-1,1))
                    values = np.unique(kbd.inverse_transform(kbd.transform(values.reshape(-1,1))).reshape(-1,9),axis=0)
                else:
                    kbd.fit(values)
                    values = np.unique(kbd.inverse_transform(kbd.transform(values)),axis=0)
                
                
                if use_random_selection is None:
                    all_values.append(values)
                    self._log2("{} candidates remaining after filtering".format(values.shape[0]))
                    distances = compute_distances(X_strides, values)
                    self._log2("------------")
                    all_distances.append(distances)
                else:
                    idx = np.random.choice(range(values.shape[0]), int(np.ceil(values.shape[0]*use_random_selection)), replace=False)
                    all_values.append(values[idx])
                    self._log2("{} candidates remaining after filtering".format(idx.shape[0]))
                    distances = compute_distances(X_strides, values[idx])
                    self._log2("------------")
                    all_distances.append(distances)
            all_params.append(k)
            all_params_id.append(values.shape[0])
        all_params_id = np.cumsum(all_params_id)
        all_params_id = np.array([[all_params_id[i],all_params_id[i+1]] 
                                  for i in range(len(all_params_id)-1)])
        all_values = np.concatenate(all_values,axis=0)
        all_distances = np.concatenate(all_distances,axis=1)
        self._log("{} candidates extracted from all kernels".format(all_values.shape[0]))
        rfc = RandomForestClassifier(n_estimators=400, ccp_alpha=0.005, max_samples=0.5).fit(all_distances, y)
        id_ft = np.where(rfc.feature_importances_>0)[0]
        self._log("Selected {} candidates".format(id_ft.shape[0]))
        self.model = RandomForestClassifier(n_estimators=400, ccp_alpha=0.015, max_samples=0.5).fit(
            all_distances[:,id_ft], y)
        id_ft = np.where(rfc.feature_importances_>0)[0]
        self._log("Used {} candidates in Forest".format(id_ft.shape[0]))
        for f_id in id_ft:
            idx = np.where((all_params_id[:,0]<=f_id)&(all_params_id[:,1]>f_id))[0][0]
            
            p = all_params[idx]
            self.shapelets.append(Convolutional_shapelet(values=all_values[f_id], 
                                                          dilation=p[1],
                                                          padding=p[0],
                                                          input_ft_id=0))
        return self
    
    
    def predict(self, X):
        X_shp = np.concatenate([shp.transform(X) for shp in self.shapelets],axis=1)
        return self.model.predict(X_shp)
            
    
    def _input_value_matrix_from_ft_kernelv2(self, X, K,):
        padding = K[0].padding
        length = K[0].length
        dilation = K[0].dilation
        n_samples, _, n_timestamps = X.shape
        n_conv = n_timestamps - ((length - 1) * dilation) + (2 * padding)
        if padding > 0:
            X_pad = np.zeros((n_samples, n_timestamps+2*padding))
            X_pad[:,padding:-padding] = X[:,0,:]
        else:
            X_pad = X[:,0,:]
        X_strides = generate_strides_2D(X_pad,length,dilation)
        X_values = []
        for i_k in range(K.shape[0]):
            idx = np.zeros((n_samples, n_conv))
            x_loc = K[i_k].get_locs(X)
            for i_x in range(n_samples):
                idx[i_x, x_loc[i_x].astype(int)] += 1
            X_values.append(X_strides[idx.astype(bool)])
        return np.concatenate(X_values, axis=0), X_strides
    
    def _input_value_matrix_from_ft_kernel(self, X, k):
        # Where in each convolution ft_func happens
        conv_idx = k.get_locs(X)
        n_timestamps = X.shape[2]
        if conv_idx.shape[0] > 0:
            max_occurences = np.max([conv_idx[i].shape[0] for i in range(X.shape[0])])
            conv_values = np.zeros((X.shape[0], max_occurences, k.length))
            for i_conv in range(X.shape[0]):
                for i_occ in range(conv_idx[i_conv].shape[0]):
                    x_idx = k._get_indexes(conv_idx[i_conv][i_occ])
                    for i_l in range(k.length):    
                        if 0<= x_idx[i_l] < n_timestamps:
                            conv_values[i_conv,i_occ,i_l] =  X[i_conv][0][x_idx[i_l]]
            #Flatten fragment matrix
            conv_values = conv_values.reshape(X.shape[0] * max_occurences, k.length).astype(float)            
            #Remove rows with only 0's
            conv_values = np.delete(conv_values, np.where((conv_values == 0).all(axis=1))[0],axis=0)
            conv_values = (conv_values - conv_values.mean(axis=-1, keepdims=True)) / (
                    conv_values.std(axis=-1, keepdims=True) + 1e-8)
            return np.unique(conv_values,axis=0)
        else:
            return None
    
    def _group_by_values(self, arr):
        sort_arr = np.argsort(arr)
        return np.array(np.split(sort_arr,
                                 np.unique(arr[sort_arr],return_index=True)[1]),
                        dtype='object')[1:]
        
    def _check_is_init(self):
        if any(self.__dict__[attribute] is None for attribute in ['n_kernels',
                                                                  'kernel_sizes',
                                                                  'id_ft']):
            raise AttributeError("Kernel attribute not initialised correctly, "
                                 "at least one attribute was set to None")
    
    def _check_array(self, X, coerce_to_numpy=True):
        if X.ndim != 3:
            raise ValueError(
                "If passed as a np.array, X must be a 3-dimensional "
                "array, but found shape: {X.shape}"
            )
        if isinstance(X, pd.DataFrame):
            if not is_nested_dataframe(X):
                raise ValueError(
                    "If passed as a pd.DataFrame, X must be a nested "
                    "pd.DataFrame, with pd.Series or np.arrays inside cells."
                )
            # convert pd.DataFrame
            if coerce_to_numpy:
                X = from_nested_to_3d_numpy(X)
        return X
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 08:52:54 2021

@author: A694772
"""
import numpy as np
import pandas as pd
from sktime.utils.data_processing import from_nested_to_3d_numpy, is_nested_dataframe
from sklearn.base import BaseEstimator, ClassifierMixin
from numba import njit, prange
from CST.base_transformers.rocket import ROCKET
from CST.factories.kernel_factories import Rocket_factory
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from CST.base_transformers.shapelets import Convolutional_shapelet

@njit(parallel=True, fastmath=True)
def compute_distances(X_strides, value_matrix):
    dist_to_X = np.zeros((X_strides.shape[0], value_matrix.shape[0]))
    for i in prange(X_strides.shape[0]):
        for j in prange(value_matrix.shape[0]):
            dist_to_X[i,j] = np.min(np.array([np.linalg.norm(X_strides[i][k]-value_matrix[j]) for k in prange(X_strides[i].shape[0])]))
    return dist_to_X

def convolution_values1D(ts, window, dilation):
    shape = (ts.size - ((window-1)*dilation), window)
    strides = np.array([ts.strides[0], ts.strides[0]*dilation])
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)    

def convolution_values2D(ts, window, dilation):
    n_rows, n_columns = ts.shape
    shape = (n_rows, n_columns - ((window-1)*dilation), window)
    strides = np.array([ts.strides[0], ts.strides[1], ts.strides[1]*dilation])
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)    

class UFShapeletClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None, id_ft=0, verbose=0, shp_len=[0.1]):
        self.id_ft = id_ft
        self.shp_len = shp_len
        self.verbose = verbose
        self.random_state = random_state
        self.shapelets = []
    
    def _log(self, message):
        if self.verbose > 0:
            print(message)
    
    def _log2(self, message):
        if self.verbose > 1:
            print(message)
    
    def fit(self, X, y, percentage_selected=0.01):
        X = self._check_array(X)
        all_values = []
        all_distances = []
        X = X[:,0,:]
        for shp_l in self.shp_len:
            X_strides = convolution_values2D(X,int(X.shape[1]*shp_l),1)
            values = np.concatenate(X_strides,axis=0)
            idx = np.random.choice(range(values.shape[0]), int(values.shape[0]*percentage_selected), replace=False)
            distances = compute_distances(X_strides, values[idx])
            all_values.extend(values[idx])
            all_distances.append(distances)
        
        all_values = np.array(all_values)
        print(all_values.shape)
        all_distances = np.concatenate(all_distances,axis=1)
        print(all_distances.shape)
        print('Learning forest ...')
        rfc = RandomForestClassifier(n_estimators=400, ccp_alpha=0.005, max_samples=0.5).fit(all_distances, y)
        self.model=rfc
        for i_val in range(all_values.shape[0]):
            self.shapelets.append(Convolutional_shapelet(values=all_values[i_val], 
                                                         dilation=1,
                                                         padding=0,
                                                         input_ft_id=0))
        print('Learned forest !')
        return self
    
    
    def predict(self, X):
        X = X[:,0,:]
        all_distances = []
        for shp_l in self.shp_len:
            s_len = int(X.shape[1]*shp_l)
            X_strides = convolution_values2D(X,s_len,1)
            values = np.array([shp.values for shp in self.shapelets if len(shp.values) == s_len])
            distances = compute_distances(X_strides, values)            
            all_distances.append(distances)
        all_distances = np.concatenate(all_distances,axis=1)
        return self.model.predict(all_distances)
            
    
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
        X_strides = convolution_values2D(X_pad,length,dilation)
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
    
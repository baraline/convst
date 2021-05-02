# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:52:44 2021

@author: Antoine
"""
import numpy as np
from numba import set_num_threads
from CST.utils.shapelets_utils import generate_strides_2D, shapelet_dist_numpy
from CST.utils.checks_utils import check_array_3D
from CST.base_transformers.minirocket import MiniRocket
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
#TODO : Add a value mapping to handle case where difference is made by raw conv value density and not location

#TODO : Implement parallelisation of candidates generation / distance computation + benchmarks
class MiniConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,  P=[100, 90, 80], n_splits=4, id_ft=0, verbose=0, n_threads=3):
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None
        self.shapelets_values = None
        self.P = P
        self.n_splits = n_splits
        self.n_threads = n_threads

    def _log(self, message):
        if self.verbose > 0:
            print(message)

    def _log2(self, message):
        if self.verbose > 1:
            print(message)

    def fit(self, X, y, use_class_weights=True):
        X = check_array_3D(X)
        set_num_threads(self.n_threads)
        #locs = (n_samples, n_kernels, n_timestamps)
        locs, dils, biases = self._init_kernels(X, y)

        groups_id, unique_groups = self._get_kernel_groups(dils, biases)
        #Maximum Shapelet extracted = |P|*n_splits*n_classes*n_grps
        self._log("Begining extraction with {} kernel groups".format(
            len(unique_groups)))
        classes = set(np.unique(y))
        n_classes = np.unique(y).shape[0]

        n_shapelets = 0
        values = {}
        for i_grp in unique_groups.keys():
            dilation = int(unique_groups[i_grp][0])
            values_grp = []
            #Sum of L for each kernel in group for all samples
            locs_conv = np.sum(
                locs[:, np.where(groups_id == i_grp)[0], :], axis=1)
            #From input space, go back to convolutional space (n_samples, n_conv, 9), sum it to get a per conv point score
            locs_conv = generate_strides_2D(
                locs_conv, 9, dilation).sum(axis=-1)
            
            #Computing splits indexes, shape (n_split, n_classes, n_idx)
            
            if all(self.n_splits <= np.bincount(y)):
                n_splt = self.n_splits
            else:
                n_splt = min(np.bincount(y))
                warnings.warn("Reduced n_split to minimum number of class sample")
            if n_splt > 1:
                sss = StratifiedShuffleSplit(n_splits=n_splt, test_size=None, train_size=1/n_splt)
                id_splits = []
                for train_index, _ in sss.split(X, y):
                    id_splits.append([train_index[np.where(y[train_index] == i_class)[0]] for i_class in np.unique(y)])
            else:
                id_splits = [[np.where(y==i_class)[0] for i_class in np.unique(y)]]
            #Compute class weight of each split
            c_w = []
            for i_split in range(n_splt):
                ys = []
                for i_class in np.unique(y):
                    ys.extend(id_splits[i_split][i_class])
                c_w.append(compute_class_weight(
                    'balanced', classes=np.unique(y[ys]), y=y[ys]))
            #Compute LC per split
            per_split_class_loc = np.zeros(
                (n_classes, n_splt, locs_conv.shape[1]))
            for i_class in classes:
                for i_split in range(n_splt):
                    per_split_class_loc[i_class, i_split, :] = locs_conv[id_splits[i_split][i_class]].sum(
                        axis=0)
                    if use_class_weights:
                        per_split_class_loc[i_class, i_split,
                                            :] *= c_w[i_split][i_class]
            for i_class in classes:
                for i_split in range(n_splt):

                    diff_other_class = np.asarray([per_split_class_loc[j, i_split, :]
                                                   for j in classes - {i_class}]).mean(axis=0)

                    D = per_split_class_loc[i_class,
                                            i_split, :] - diff_other_class
                    loc = np.asarray(
                        [np.abs(D-np.percentile(D, p)).argmin() for p in self.P])
                    id_x = [
                        np.argmax(locs_conv[id_splits[i_split][i_class], i]) for i in loc]
                    for i, ix in enumerate(id_x):
                        values_grp.append(X[id_splits[i_split][i_class][ix], 0, np.array(
                            [loc[i]+j*dilation for j in range(9)])])

            values_grp = np.asarray(values_grp)
            self._log2("Got {} candidates for grp {}".format(
                values_grp.shape[0], i_grp))
            if values_grp.shape[0] > 0:
                values_grp = (values_grp - values_grp.mean(axis=-1, keepdims=True)) / (
                    values_grp.std(axis=-1, keepdims=True) + 1e-8)
                if not np.all(values_grp.reshape(-1, 1) == values_grp.reshape(-1, 1)[0]):
                    kbd = KBinsDiscretizer(n_bins=9, strategy='uniform',dtype=np.float32).fit(
                        values_grp.reshape(-1, 1))
                    values_grp = np.unique(kbd.inverse_transform(
                        kbd.transform(values_grp.reshape(-1, 1))).reshape(-1, 9), axis=0)
                else:
                    values_grp = np.unique(values_grp, axis=0)
                n_shapelets += values_grp.shape[0]
                values.update({i_grp: values_grp})
                self._log("Extracted {} shapelets for grp {}/{}".format(
                    values_grp.shape[0], i_grp, len(unique_groups.keys())))

        self.shapelets_params = {
            i_grp: unique_groups[i_grp] for i_grp in values.keys()}
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
                                                                                 i, len(
                                                                                     self.shapelets_params),
                                                                                 self.shapelets_values[i_grp].shape[0]))
            dilation, _ = self.shapelets_params[i_grp]
            X_strides = self._get_X_strides(X, 9, dilation, 0)
            d = shapelet_dist_numpy(X_strides, self.shapelets_values[i_grp])
            distances[:, prev:prev+d.shape[1]] = d
            prev += d.shape[1]
        return distances

    def _get_kernel_groups(self, kernels_dilations, kernels_bias):
        kernels_bias = np.array([b >= 0 for b in kernels_bias]).astype(int)
        groups_params = np.array([[kernels_dilations[i],
                                   kernels_bias[i]]
                                  for i in range(kernels_dilations.shape[0])], dtype=np.int32)

        groups_id = np.zeros(kernels_dilations.shape[0])
        unique_groups = np.unique(groups_params, axis=0)
        unique_groups = {i: unique_groups[i]
                         for i in range(unique_groups.shape[0])}

        for i in unique_groups.keys():
            groups_id[np.where(
                (groups_params == unique_groups[i]).all(axis=1))[0]] = i
        return groups_id, unique_groups

    def _get_X_strides(self, X, length, dilation, padding):
        n_samples, _, n_timestamps = X.shape
        if padding > 0:
            X_pad = np.zeros((n_samples, n_timestamps+2*padding))
            X_pad[:, padding:-padding] = X[:, self.id_ft, :]
        else:
            X_pad = X[:, self.id_ft, :]
        X_strides = generate_strides_2D(X_pad, length, dilation)
        X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
            X_strides.std(axis=-1, keepdims=True) + 1e-8)
        return X_strides

    def _init_kernels(self, X, y):
        self._log("Performing MiniRocket Transform")
        m = MiniRocket().fit(X)
        ft, locs = m.transform(X, return_locs=True)
        self._log(
            "Performing kernel selection with {} kernels".format(locs.shape[1]))
        """
        ft_selector = SelectFromModel(RandomForestClassifier(max_features=0.85,
                                                             max_samples=0.85,
                                                             ccp_alpha=0.02,
                                                             n_jobs=None)).fit(ft,y)
        """
        ft_selector = SelectFromModel(
            DecisionTreeClassifier(ccp_alpha=0.005)).fit(ft, y)
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=int)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n += 84*num_features_per_dilation[i]

        i_kernels = np.where(ft_selector.get_support())[0]
        self.n_kernels = i_kernels.shape[0]
        self._log("Finished kernel selection with {} kernels".format(
            i_kernels.shape[0]))
        return locs[:, i_kernels], dils[i_kernels], biases[i_kernels]

    def _check_is_fitted(self):
        if any(self.__dict__[attribute] is None for attribute in ['shapelets_values',
                                                                  'shapelets_params']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")

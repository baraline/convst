# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:43:20 2021

@author: A694772
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:52:44 2021

@author: Antoine
"""

# TODO : refactor as a classifier
# TODO : Docs !
# TODO : Implement parallelisation of candidates generation / distance computation + benchmarks
# TODO : Implement parameter to change length of kernel/shapelet

import numpy as np
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from CST.base_transformers.minirocket import MiniRocket
from CST.utils.checks_utils import check_array_3D
from CST.utils.shapelets_utils import generate_strides_2D, shapelet_dist_numpy, generate_strides_1D
from numba import set_num_threads, njit, prange

class ConvolutionalShapeletTransformer_tree(BaseEstimator, TransformerMixin):
    def __init__(self,  P=80, n_trees=200, max_ft=1.0, id_ft=0, use_class_weights=True,
                 verbose=0, n_bins=9, n_threads=3, random_state=None):
        """
        Initialize the Convolutional Shapelet Transform (CST)

        Parameters
        ----------
        P : array of int, optional
            Percentile used in the shapelet extraction process.
            The default is 80.
        n_trees : int or float, optional
        
        use_class_weights : bool, optional
        
        id_ft : int, optional
            Identifier of the feature on which the transform will be performed.
            The default is 0.
        verbose : int, optional
            Verbose parameter, higher values will output more logs. The default is 0.
        n_bins : int, optional
            Number of bins used in the candidates discretization. The default is 9.
        n_threads : int, optional
            Number of numba thread used. The default is 3.
        use_kernel_grouping : bool, optional
            Wheter or not to enable kernel grouping based on dilation and bias parameter.
            The default is True.
        random_state : int, optional
            Random state setter. The default is None.

        Returns
        -------
        None.

        """
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None
        self.shapelets_values = None
        self.P = P
        self.n_trees = n_trees
        self.use_class_weights=use_class_weights
        self.max_ft = max_ft
        self.n_bins = n_bins
        self.n_threads = n_threads
        self.random_state = random_state

    def _log(self, message):
        if self.verbose > 0:
            print(message)

    def _log2(self, message):
        if self.verbose > 1:
            print(message)

    def fit(self, X, y, use_class_weights=True):
        """

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input data containing time series (tested with dtype np.float32), the algorithm
            will only process feature indicated by attribute id_ft.
        y : array, shape = (n_samples)
            Associated classes of the input time series
        use_class_weights : TYPE, optional
            Whether or not to balance computation based on the number of samples
            of each class.

        Returns
        -------
        ConvolutionalShapeletTransformer
            Fitted instance of self.

        """
        X = check_array_3D(X)
        set_num_threads(self.n_threads)
        #L = (n_samples, n_kernels, n_timestamps)
        # Kernel selection is performed in this function
        L, dils, biases, tree_splits = self._generate_inputs(X, y)
        shapelets = {}
        n_shp = 0
        for i_split in range(len(tree_splits)):
            self._log2(
                "Processing split {}/{} ...".format(i_split, len(tree_splits)))
            x_index, y_split, k_id = tree_splits[i_split]
            
            @njit
            def _generate_strides_2d(X, window, dilation)
                n_rows, n_columns = X.shape
                shape = (n_rows, n_columns - ((window-1)*dilation), window)
                strides = np.array([X.strides[0], X.strides[1], X.strides[1]*dilation])
                return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)  
            
            @njit(fastmath=True, cache=True)
            def _process_node(X, y, L, dilation):
                classes = np.unique(y)
                n_classes = classes.shape[0]
                
                #To test
                Lp = generate_strides_2D(L, 9, dilation).sum(axis=-1)
               
                #to replace
                c_w = compute_class_weight('balanced', classes=classes, y=y_split)
                
                LC = np.zeros((n_classes, Lp.shape[1]))
                for i_class in prange(n_classes):
                    LC[i_class] = c_w[i_class] * Lp[
                        np.where(y_split == i_class)[0]].sum(axis=0)
    
                candidates_grp = np.zeros(n_candidates,9)
                for i_class in prange(n_classes):
                    if LC.sum() > 0:
                        D = LC[i_class] - LC[(i_class+1) % 2]
                        id_D = np.where(D >= np.percentile(D, self.P))[0]
                        # To Replace
                        regions = self._get_regions(id_D)
                        for i_r in prange(regions.shape[0]):
                            LC_region = LC[i_class, regions[i_r]]
                            id_max_region = regions[i_r][LC_region.argmax()]
                            x_index = np.argmax(
                                Lp[np.where(y_split == i_class)[0], id_max_region])
                            candidates_grp.append(X[np.where(y_split == i_class)[0][x_index], 0, np.array(
                                [id_max_region+j*dilation for j in prange(9)])])
                
                return candidates_grp = np.asarray(candidates_grp)
            
            if candidates_grp.shape[0] > 0:
                if dilation in shapelets:
                    shapelets[dilation] = np.concatenate(
                        (shapelets[dilation], candidates_grp), axis=0)
                else:
                    shapelets.update({dilation: candidates_grp})

        for dil in shapelets.keys():
            candidates_grp = shapelets[dil]
            if candidates_grp.shape[0] > 0:
                candidates_grp = (candidates_grp - candidates_grp.mean(axis=-1, keepdims=True)) / (
                    candidates_grp.std(axis=-1, keepdims=True) + 1e-8)
                if not np.all(candidates_grp.reshape(-1, 1) == candidates_grp.reshape(-1, 1)[0]):
                    kbd = KBinsDiscretizer(n_bins=self.n_bins, strategy='uniform', dtype=np.float32).fit(
                        candidates_grp.reshape(-1, 1))
                    candidates_grp = np.unique(kbd.inverse_transform(
                        kbd.transform(candidates_grp.reshape(-1, 1))).reshape(-1, 9), axis=0)
                else:
                    candidates_grp = np.unique(candidates_grp, axis=0)
                shapelets[dil] = candidates_grp
                n_shp += candidates_grp.shape[0]

        self.shapelets_values = shapelets
        self.n_shapelets = n_shp
        return self

    def transform(self, X):
        """
        Transform input time series into Shapelet distances

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input data containing time series (tested with dtype np.float32), the algorithm
            will only process feature indicated by attribute id_ft.

        Returns
        -------
        distances : array, shape = (n_samples, n_shapelets)
            Shapelet distance to all samples

        """
        self._check_is_fitted()
        X = check_array_3D(X)
        distances = np.zeros((X.shape[0], self.n_shapelets))
        prev = 0
        for i, dil in enumerate(self.shapelets_values.keys()):
            self._log("Transforming for dilation {} ({}/{}) with {} shapelets".format(
                dil, i, len(self.shapelets_values), len(self.shapelets_values[dil])))
            if len(self.shapelets_values[dil]) > 0:
                dilation = dil
                X_strides = self._get_X_strides(X, 9, dilation, 0)
                d = shapelet_dist_numpy(
                    X_strides, self.shapelets_values[dil])
                distances[:, prev:prev+d.shape[1]] = d
                prev += d.shape[1]
        return distances

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

    def _get_regions(self, indexes):
        regions = []
        region = []
        for i in range(indexes.shape[0]-1):
            region.append(indexes[i])
            if indexes[i] != indexes[i+1]-1:
                regions.append(region)
                region = []
        if len(region) > 0:
            regions.append(region)
        return regions

    def _generate_inputs(self, X, y):
        """


        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        self._log("Performing MiniRocket Transform")
        m = MiniRocket().fit(X)
        ft, locs = m.transform(X, return_locs=True)
        self._log(
            "Performing kernel selection with {} kernels".format(locs.shape[1]))
        if self.use_class_weights:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        class_weight='balanced')
        else:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft)
        rf.fit(ft, y)
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=int)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n += 84*num_features_per_dilation[i]

        tree_splits = []

        def extract_tree_splits(tree, features,  y,):
            tree_ = tree.tree_
            x_id = np.asarray(range(features.shape[0]))

            def recurse(node, depth, x_id):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    ft_id = tree_.feature[node]
                    threshold = tree_.threshold[node]
                    L = x_id[np.where(features[x_id, ft_id] <= threshold)[0]]
                    R = x_id[np.where(features[x_id, ft_id] > threshold)[0]]
                    y_node = np.zeros(x_id.shape[0], dtype=int)
                    y_node[L.shape[0]:] += 1
                    recurse(tree_.children_left[node], depth + 1,
                            x_id[np.where(features[x_id, ft_id] <= threshold)[0]])
                    recurse(tree_.children_right[node], depth + 1,
                            x_id[np.where(features[x_id, ft_id] > threshold)[0]])
                    tree_splits.append(
                        [np.concatenate((L, R), axis=0), y_node, ft_id])
            recurse(0, 1, x_id)

        for dt in rf.estimators_:
            extract_tree_splits(dt, ft, y)

        self._log("Extracted {} splits".format(len(tree_splits)))
        return locs, dils, biases, tree_splits

    def _check_is_fitted(self):
        """


        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if any(self.__dict__[attribute] is None for attribute in ['shapelets_values']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")

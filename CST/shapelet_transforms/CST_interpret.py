# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:50:24 2021

@author: A694772
"""
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
from numba import set_num_threads

class ConvolutionalShapeletTransformer_interpret(BaseEstimator, TransformerMixin):
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
        self.m = MiniRocket().fit(X)
        ft, L = self.m.transform(X, return_locs=True)
        if self.use_class_weights:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        class_weight='balanced',
                                        ccp_alpha=0.00,
                                        n_jobs=self.n_threads)
        else:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        ccp_alpha=0.00,
                                        n_jobs=self.n_threads)
            
        dilations, num_features_per_dilation, biases = self.m.parameters
        dils = np.zeros(biases.shape[0], dtype=int)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n += 84*num_features_per_dilation[i]
        
        self.rf = rf.fit(ft, y)
        dt_info = []
        for i, dt in enumerate(self.rf.estimators_):
            node_values = []
            node_location = []
            n_info = self.tree_nodes_split(ft, y, dt)
            for i_ft in dt.tree_.feature:
                if i_ft == _tree.TREE_UNDEFINED:
                    node_values.append([])
                    node_location.append([])
                else:
                    for j in range(len(n_info)):
                        if n_info[j][2] == i_ft:
                            x_index, y_split, k_id = n_info[j]
                            
                            n_v, n_l = self.process_node(X[x_index],
                                                  y_split,
                                                  L[x_index,k_id, :],
                                                  dils[n_info[j][2]],
                                                  self.P)
                            node_values.append(n_v)
                            node_location.append(n_l)

            dt_info.append([node_values, node_location])
        
        self.dt_info = dt_info
        return self
    
    def process_node(self, X, y, L, dilation, P):
        classes = np.unique(y)
        n_classes = classes.shape[0]

        Lp = generate_strides_2D(L, 9, dilation).sum(axis=-1)
        
        c_w = compute_class_weight('balanced', classes=classes, y=y)
        
        LC = np.zeros((n_classes, Lp.shape[1]))
        for i_class in classes:
            LC[i_class:] = c_w[i_class] * Lp[
                np.where(y == i_class)[0]].sum(axis=0)

        candidates_values = []
        candidates_locations = []
        for i_class in classes:
            if LC.sum() > 0:
                D = LC[i_class] - LC[(i_class+1) % 2]
                id_D = np.where(D >= np.percentile(D, self.P))[0]
                # A region is a set of following indexes
                regions = self._get_regions(id_D)
                for i_r in range(len(regions)):
                    LC_region = LC[i_class, regions[i_r]]
                    id_max_region = regions[i_r][LC_region.argmax()]
                    x_index = np.argmax(
                        Lp[np.where(y == i_class)[0], id_max_region])
                    candidates_values.append(X[np.where(y == i_class)[0][x_index],
                                            0,
                                            np.array([id_max_region+j*dilation 
                                                      for j in range(9)])
                                            ])
                    candidates_locations.append(np.array([id_max_region+j*dilation 
                                                      for j in range(9)]))
        candidates_locations = np.asarray(candidates_locations)
        candidates_values = np.asarray(candidates_values)
        candidates_values = (candidates_values - candidates_values.mean(axis=-1, keepdims=True)) / (
                    candidates_values.std(axis=-1, keepdims=True) + 1e-8)
        return candidates_values, candidates_locations.astype(int)
            
        
    def transform(self, X, y=None):
        X = check_array_3D(X)
        n_areas = np.zeros((X.shape[0],X.shape[2]),dtype=int)
        X_rkt, locs = self.m.transform(X, return_locs=True)
        for i in range(X.shape[0]):
            n_areas[i] += self.interpet_one_sample(X_rkt[i], y, X[i])
        return n_areas
        
    
    def interpet_one_sample(self, x, y, ts):
        n_areas = np.zeros(ts.shape[1], dtype=int)
        for i, dt in enumerate(self.rf.estimators_):
            for i_loc in self.interpret_one_tree(x, y, ts, dt, self.dt_info[i]):
                n_areas[i_loc] += 1
        return n_areas

    def interpret_one_tree(self, x, y, ts, dt, dt_info):
        node_indicator = dt.decision_path([x])
        sample_id = 0
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        
        d = np.asarray([self.min_d(dt_info[0][i],dt_info[1][i],ts) for i in node_index[:-1]],dtype='object').flatten()
        return d
        
    def min_d(self, s, l, x):
        res = np.zeros((s.shape[0],9),dtype=int)
        for i in range(s.shape[0]):
            d=int(l[i][1]-l[i][0])            
            strd_x = generate_strides_1D(x, 9, d)
            srtd_x = (strd_x - strd_x.mean(axis=-1,keepdims=True))/strd_x.std(axis=-1,keepdims=True)
            dist = np.abs(strd_x - s[i]).sum(axis=1)            
            i_l = np.argmin(dist)
            res[i,:] = np.array([i_l + j*d for j in range(9)])
        return res
    
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


    def tree_nodes_split(self, ft, y, dt):
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

        extract_tree_splits(dt, ft, y)
        return tree_splits

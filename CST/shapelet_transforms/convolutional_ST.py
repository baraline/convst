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
import numpy as np
from numba import set_num_threads
from CST.utils.shapelets_utils import generate_strides_2D, shapelet_dist_numpy
from CST.utils.checks_utils import check_array_3D
from CST.base_transformers.minirocket import MiniRocket
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

#TODO : Docs !
#TODO : Implement parallelisation of candidates generation / distance computation + benchmarks
#TODO : Implement parameter to change length of kernel/shapelet 
#TODO : Make all argument non positional
class ConvolutionalShapeletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,  P=[100, 95, 90, 85, 80], n_splits=10, id_ft=0,
                 verbose=0, n_bins=9, n_threads=3, use_kernel_grouping=True,
                 random_state=None):
        """
        Initialize the Convolutional Shapelet Transform (CST)
        
        Parameters
        ----------
        P : array of int, optional
            Percentiles used in the shapelet extraction process.
            The default is [100, 95, 90, 85, 80].
        n_splits : int, optional
            Number of stratified shuffle split performed for each extraction round.
            The default is 10.
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
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.n_threads = n_threads
        self.use_kernel_grouping = use_kernel_grouping
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
        #Kernel selection is performed in this function
        L, dils, biases = self._generate_inputs(X, y)
        
        #Grouping kernel, use_kernel_grouping attribute is used here
        groups_id, groups_param = self._group_kernels(dils, biases)
        
        self._log("Begining extraction with {} kernels".format(len(groups_param)))
        classes = set(np.unique(y))
        n_classes = np.unique(y).shape[0]
        n_shapelets = 0
        shapelets = {}
        for i_grp in groups_param.keys():
            dilation = int(groups_param[i_grp][0])
            candidates_grp = []
            #Sum of L for each kernel in group for all samples
            Lp = np.sum(L[:, np.where(groups_id == i_grp)[0], :], axis=1)
            #From input space, go back to convolutional space (n_samples, n_conv, 9),
            #sum it to get a per conv point score as (n_samples, n_conv)
            Lp = generate_strides_2D(Lp, 9, dilation).sum(axis=-1)

            #Computing splits indexes, shape (n_split, n_classes, n_idx)
            n_splt, id_splits = self._stratifiedShuffleSpliter(X, y)
            #Compute class weight of each split
            c_w = self._split_classweights(y ,n_splt, id_splits)
            #Compute LC for all splits
            per_split_LC = self._compute_LC_per_split(Lp, id_splits, n_splt,
                                                      n_classes, classes,
                                                      c_w, use_class_weights)
            #Extract candidates for all splits
            candidates_grp = self._extract_candidates(X, Lp, per_split_LC, 
                                                      id_splits, n_splt, 
                                                      classes, dilation)
            self._log2("Got {} candidates for kernel {}".format(candidates_grp.shape[0], i_grp))
            
            candidates_grp = self._remove_similar(candidates_grp)
            n_shapelets += candidates_grp.shape[0]
            shapelets.update({i_grp: candidates_grp})
            
            self._log("Extracted {} shapelets for kernel {}/{}".format(
                candidates_grp.shape[0], i_grp, len(groups_param.keys())))

        self.shapelets_params = {i_grp: groups_param[i_grp] for i_grp in shapelets.keys()}
        self.shapelets_values = shapelets
        self._log("Extracted a total of {} shapelets".format(n_shapelets))
        self.n_shapelets = n_shapelets
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
        for i, i_grp in enumerate(self.shapelets_params.keys()):
            self._log("Transforming for kernel {} ({}/{}) with {} shapelets".format(self.shapelets_params[i_grp],
                                                                                 i, len(
                                                                                     self.shapelets_params),
                                                                                 self.shapelets_values[i_grp].shape[0]))
            dilation, _ = self.shapelets_params[i_grp]
            X_strides = self._get_X_strides(X, 9, dilation, 0)
            d = shapelet_dist_numpy(X_strides, self.shapelets_values[i_grp])
            distances[:, prev:prev+d.shape[1]] = d
            prev += d.shape[1]
        return distances


    def _remove_similar(self, candidates_grp, strategy='uniform'):
        """
        Apply discretization to candidates with n_bins attributes following
        the strategy in parameter. After that, keep only unique candidates

        Parameters
        ----------
        candidates_grp : array, shape = (n_candidates, kernel_length)
            DESCRIPTION.
        strategy : string, optional
            Strategy used for the discritization using KBinsDiscretizer.
            The default is 'uniform'.

        Returns
        -------
        candidates_grp : array, shape = (n_discretized_candidates, kernel_length)
            Discretized candidates with removed duplicates

        """
        if candidates_grp.shape[0] > 0:
            candidates_grp = (candidates_grp - candidates_grp.mean(axis=-1, keepdims=True)) / (
                candidates_grp.std(axis=-1, keepdims=True) + 1e-8)
            if not np.all(candidates_grp.reshape(-1, 1) == candidates_grp.reshape(-1, 1)[0]):
                kbd = KBinsDiscretizer(n_bins=self.n_bins, strategy=strategy, dtype=np.float32).fit(
                    candidates_grp.reshape(-1, 1))
                candidates_grp = np.unique(kbd.inverse_transform(
                    kbd.transform(candidates_grp.reshape(-1, 1))).reshape(-1, 9), axis=0)
            else:
                candidates_grp = np.unique(candidates_grp, axis=0)
        return candidates_grp

    def _extract_candidates(self, X, Lp, per_split_LC, id_splits, n_splt, classes, dilation):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Lp : TYPE
            DESCRIPTION.
        per_split_LC : TYPE
            DESCRIPTION.
        id_splits : TYPE
            DESCRIPTION.
        n_splt : TYPE
            DESCRIPTION.
        classes : TYPE
            DESCRIPTION.
        dilation : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        candidates_grp = []
        for i_class in classes:
            for i_split in range(n_splt):
                #Mean of other classes
                D = np.asarray([per_split_LC[j, i_split, :] for j in classes - {i_class}]).mean(axis=0)
                D = per_split_LC[i_class,i_split, :] - D
                
                loc = np.asarray([np.abs(D-np.percentile(D, p)).argmin() for p in self.P])
                x_index = [np.argmax(Lp[id_splits[i_split][i_class], i]) for i in loc]
                for i, ix in enumerate(x_index):
                    candidates_grp.append(X[id_splits[i_split][i_class][ix], 0, np.array(
                        [loc[i]+j*dilation for j in range(9)])])
        return np.asarray(candidates_grp)

    def _compute_LC_per_split(self, Lp, id_splits, n_splt, n_classes, classes, c_w, use_class_weights):
        """
        

        Parameters
        ----------
        Lp : TYPE
            DESCRIPTION.
        id_splits : TYPE
            DESCRIPTION.
        n_splt : TYPE
            DESCRIPTION.
        n_classes : TYPE
            DESCRIPTION.
        classes : TYPE
            DESCRIPTION.
        c_w : TYPE
            DESCRIPTION.
        use_class_weights : TYPE
            DESCRIPTION.

        Returns
        -------
        per_split_LC : TYPE
            DESCRIPTION.

        """
        per_split_LC = np.zeros((n_classes,  n_splt, Lp.shape[1]))
        for i_class in classes:
            for i_split in range(n_splt):
                per_split_LC[i_class, i_split, :] = Lp[id_splits[i_split][i_class]].sum(axis=0)
                if use_class_weights:
                    per_split_LC[i_class, i_split, :] *= c_w[i_split][i_class]
        return per_split_LC
        

    def _split_classweights(self, y, n_splt, id_splits):
        """
        

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        n_splt : TYPE
            DESCRIPTION.
        id_splits : TYPE
            DESCRIPTION.

        Returns
        -------
        c_w : TYPE
            DESCRIPTION.

        """
        c_w = []
        for i_split in range(n_splt):
            ys = []
            for i_class in np.unique(y):
                ys.extend(id_splits[i_split][i_class])
            c_w.append(compute_class_weight(
                'balanced', classes=np.unique(y[ys]), y=y[ys]))
        return c_w 
    
    def _stratifiedShuffleSpliter(self, X, y):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if all(self.n_splits <= np.bincount(y)):
            n_splt = self.n_splits
        else:
            n_splt = min(np.bincount(y))
            s = "Reduced n_split to minimum number of class sample ({})".format(n_splt)
            warnings.warn(s)
        if n_splt > 1:
            sss = StratifiedShuffleSplit(
                n_splits=n_splt, test_size=None, train_size=1/n_splt)
            id_splits = []
            for train_index, _ in sss.split(X, y):
                id_splits.append([train_index[np.where(y[train_index] == i_class)[
                                 0]] for i_class in np.unique(y)])
        else:
            id_splits = [[np.where(y == i_class)[0]
                          for i_class in np.unique(y)]]
        return n_splt, id_splits
    
    def _group_kernels(self, kernels_dilations, kernels_bias):
        """
        Depending on attribute use_kernel_grouping, this will either produce
        groups of kernels that share the same dilation parameter and either positive
        or negative bias, or it will output each kernel in its own group.

        Parameters
        ----------
        kernels_dilations : array, shape = (n_kernels)
            Array containing the dilation parameter of all kernels
        kernels_bias : array, shape = (n_kernels)
            Array containing the bias parameter of all kernels

        Returns
        -------
        array, shape = (n_kernels)
            The group identifier of each individual kernel
        dictionnary, shape = (n_groups,2)
            The groups parameters
        """
        kernels_bias = np.array([b >= 0 for b in kernels_bias]).astype(int)
        groups_params = np.array([[kernels_dilations[i],
                                   kernels_bias[i]]
                                  for i in range(kernels_dilations.shape[0])], dtype=np.int32)
        if not self.use_kernel_grouping:
            return np.array(range(kernels_dilations.shape[0])), {i : groups_params[i] for i in range(kernels_dilations.shape[0])}
        else:
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
        m = MiniRocket(random_state=self.random_state).fit(X)
        ft, locs = m.transform(X, return_locs=True)
        self._log(
            "Performing kernel selection with {} kernels".format(locs.shape[1]))
        ft_selector = SelectFromModel(
            DecisionTreeClassifier()).fit(ft, y)
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
        """
        

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if any(self.__dict__[attribute] is None for attribute in ['shapelets_values',
                                                                  'shapelets_params']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")

# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:17:04 2021

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
import numpy as np
from numba import set_num_threads, prange, njit
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
        self.shapelets_dilations = None
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
        self._log("Begining extraction with {} kernel groups".format(len(groups_param)))
        
        self.shapelets_values = _fit(X, y, L, dils, biases, groups_id, groups_param)
        self.shapelets_dilations = np.array([groups_param[i_grp]
                                            for i_grp in range(self.shapelets_values.shape[0])])
         
        self._log("Extracted a total of {} shapelets".format(self.shapelets_values.size//self.shapelets_values.shape[2]))
        self.n_shapelets = n_shapelets
        return self

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
        distances = _transform(X, self.shapelets_values, self.shapelets_dilations)
        return distances
    
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
                                                                  'shapelets_dilations']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")

    
@njit(
    cache=True,
    parallel=True,
    fastmath=True
)
def _fit(X, y, L, dils, biases, groups_id, groups_param, P, n_bins, n_splits):
    n_groups = np.unique(groups_id).shape[0]
    candidates = np.zeros((n_groups, P.shape[0]*np.unique(y).shape[0]*n_splits, 9), 
                          dtype=np.float32)
    for i in prange(n_groups):
        candidates[i,:,:] += _fit_one_group(X, y, L, groups_param[i,0], P, n_bins, n_splits)
    return candidates
     
@njit(
    cache=True,
    parallel=True,
    fastmath=True
)
def _transform(X, shapelets_values, shapelets_dilations):
    distances = np.zeros((X.shape[0],shapelets_values.shape[0]*shapelets_values.shape[1]))
    for i in prange(shapelets_values.shape[0]):
        distances[i,i*shapelets_values.shape[1]:(i+1)*shapelets_values.shape[1]] += _transform_one_group(X, shapelets_values[i], shapelets_dilations[i])
    return distances

@njit(
    cache=True,
    fastmath=True
)       
def _fit_one_group(X, y, ):
    pass

@njit(
    cache=True,
    fastmath=True
)       
def _transform_one_group(X, values, dil):
    pass
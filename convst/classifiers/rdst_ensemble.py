# -*- coding: utf-8 -*-

import numpy as np

from joblib import Parallel

from convst.transformers import R_DST
from sklearn.utils import resample, shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import delayed
from sklearn.decomposition import PCA
from sklearn.utils import resample, shuffle
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

# Most of the following have been developed for the ATM case using a binary classification.
# Multi-class might not be supported for all functions and some would need to be adapted.


from datetime import datetime
def _log_time(msg):
    print("{} - {}".format(str(datetime.now()), msg))

# Shapelet transformation in not included in base tree but only in forest builder
# If you wish to use this class by itself you must perform shapelet transform first,
# or build a pipeline
class _RotationModel(BaseEstimator, TransformerMixin):
    def __init__(
            self,            
            rot_groups=10,
            rot_components=10,
            PCA_solver='auto',
            n_shapelets=1000,
            shapelet_sizes=[0.05],
            random_state=None,
        ):
        """
        r_sample : float, optional
            Proportion of sample for each resample for PCA transforms.
            The default is 0.5.
        rot_groups : int, optional
            Number of PCA transform used on the input features. The number of 
            features per PCA transform is equal to n_features/rot_groups.
            The default is 5.
        rot_components : int or None, optional
            The number of component to fit for each PCA. None will be equal to
            min(n_samples, n_features) at each fit. The default is 20.

        Returns
        -------
        None.

        """
        if not isinstance(rot_groups, int):
            raise ValueError('rot_groups parameter should be superior to 1 and an int')
        self.rot_groups = rot_groups
        if rot_components < 1 or not isinstance(rot_components, int):
            raise ValueError('rot_components parameter should be superior to 1 and an int')
        self.rot_components = rot_components
        self.PCA_solver = PCA_solver
        self.random_state = random_state
        #TODO : make getter setter linked to rdst
        self.n_shapelets=n_shapelets
        self.shapelet_sizes=shapelet_sizes
        self.rdst = R_DST(
            n_shapelets=n_shapelets, shapelet_sizes=shapelet_sizes,
            random_state=None
        )
        
    def pca_transform(self, X):
        """
        Transform the input data with the previously fitted PCAs

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            Transformed time series data.

        Returns
        -------
        X_new : array, shape=(n_samples, n_components*rot_groups)
            PCA transformed data.

        """
        X_new = np.zeros((X.shape[0], self.output_shape), dtype=np.float64)
        
        # Index counter for affecting PCA components
        shp = 0
        for i in range(self.rot_groups):
            x = self.pcas[i].transform(X[:, self.r_features[i]])
            X_new[:, shp:shp+x.shape[1]] = x
            shp += x.shape[1]
        return X_new

    def pca_fit(self, X):
        """
        

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            Transformed time series data.
        y : array, shape=(n_samples)
            Class of each life cycle.

        Returns
        -------
        self
        
        """
        #create feature partitions
        
        id_features = np.arange(X.shape[1])
        id_features = shuffle(id_features, random_state=self.random_state)
        
        step = X.shape[1]//self.rot_groups
        id_loop = np.arange(0, X.shape[1]+step, step)
        
        pcas = []
        self.output_shape = 0
        self.r_features = []

        for i in range(self.rot_groups):
            i_ft = id_features[id_loop[i]:id_loop[i+1]]
            self.r_features.append(i_ft)
            X_r = X[:, i_ft]
            n_compo = np.min([self.rot_components, X_r.shape[0], X_r.shape[1]])
            pca = PCA(
                n_components=n_compo, svd_solver=self.PCA_solver,
                random_state=self.random_state
            ).fit(X_r)
            self.output_shape += pca.components_.shape[0]
            pcas.append(pca)
        self.pcas = pcas
        return self

    def fit(self, X, y):
        """
        Fit the PCAs with input data, transform it, and then build the tree. A
        survival model is then learned in each leaf.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Transformed time series data.
        y : array, shape=(n_samples)
            Class of each life cycle
        Returns
        -------
        self

        """
        print(np.bincount(y))
        self.rdst.fit(X, y)
        X = self.rdst.transform(X)
        if self.rot_groups > 0:
            self.pca_fit(X)
        return self

            
    def transform(self, X):
        """
        Predict the class of each sample.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Time series data.

        Raises
        ------
        ValueError
            If a sample ends up in more than one leaf, we have a problem.

        Returns
        -------
        array, shape=(n_samples)
            Predicted class.

        """
        #check is fitted
        if self.rot_groups > 0:
            X =  self.pca_transform(X)
        return self.rdst.transform(X)
        
        
def _parallel_fit(
    X, y, 
    n_shapelets,
    shapelet_sizes,
    rot_groups,
    rot_components,
    PCA_solver,
    random_state,
):
    return _RotationModel(
        rot_groups=rot_groups,
        rot_components=rot_components,
        PCA_solver=PCA_solver,
        n_shapelets=n_shapelets,
        shapelet_sizes=shapelet_sizes,
        random_state=random_state,
    ).fit(X, y)


def _parallel_transform(X, model):
    return model.transform(X)

class R_DST_Ensemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=10,
        n_shapelets_per_estimator=1000,
        shapelet_sizes=[0.05],
        rot_groups=10,
        rot_components=10,
        n_samples=0.5,
        PCA_solver='auto',
        n_jobs=1,
        backend="processes",
        base_classifier=None,
        random_state=None,
        verbose=False
    ):
        self.rot_groups=rot_groups
        self.rot_components=rot_components
        self.PCA_solver=PCA_solver
        self.n_shapelets_per_estimator=n_shapelets_per_estimator
        self.shapelet_sizes=shapelet_sizes
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.backend=backend
        if base_classifier is None:
            dt = DecisionTreeClassifier(ccp_alpha=0.005)
            self.base_classifier = dt
        else:
            self.base_classifier=base_classifier
        
        self.random_state = random_state
        self.verbose=verbose
        self.n_samples=n_samples
        
    def fit(self, X, y):
        iix = np.arange(X.shape[0])
        X_idx = [
            resample(
                iix, replace=False, stratify=y,
                n_samples=int(X.shape[0]*self.n_samples), 
                random_state=self.random_state
            )    
        for i in range(self.n_estimators)]
        models = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer=self.backend,
        )(
            delayed(_parallel_fit)(
                X[X_idx[i]],
                y[X_idx[i]],  
                self.n_shapelets_per_estimator,
                self.shapelet_sizes,
                self.rot_groups,
                self.rot_components,
                self.PCA_solver,
                self.random_state,
            )
            for i in range(self.n_estimators)
        )
        self.models = models
        X_new = self.transform(X)
        self.base_classifier.fit(X_new, y)
        return self


    def transform(self, X):
        X_new = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer=self.backend,
        )(
            delayed(_parallel_transform)(
                X,
                self.models[i]
            )
            for i in range(self.n_estimators)
        )
        #n_samples, n_models
        X_new = np.asarray(X_new).reshape(X.shape[0], -1)
        print(X_new.shape)
        return X_new
        
    def predict(self, X):
        X_new = self.transform(X)
        return self.base_classifier.predict(X_new)


    
        
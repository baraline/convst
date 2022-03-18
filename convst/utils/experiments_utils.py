# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:16:32 2022

@author: a694772
"""
import numpy as np

from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.utils import resample

from timeit import default_timer as timer

from convst.utils.dataset_utils import load_sktime_arff_file_resample_id

class stratified_resample:
    """
    A random resampler used as a splitter for sklearn cross validation tools.

    Parameters
    ----------
    n_splits : int
        Number of cross validation step planed.
    n_samples_train : int
        Number of samples to include in the randomly generated 
        training sets.


    """
    def __init__(self, n_splits, n_samples_train):
        
        self.n_splits=n_splits
        self.n_samples_train=n_samples_train
        
    def split(self, X, y=None, groups=None):
        """
        

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Time series data to split
        y : ignored

        groups : ignored

        Yields
        ------
        idx_Train : array, shape(n_samples_train)
            Index of the training data in the original dataset X.
        idx_Test : array, shape(n_samples_test)
            Index of the testing data in the original dataset X.

        """
        idx_X = np.asarray(range(X.shape[0]))
        for i in range(self.n_splits):
            if i == 0:
                idx_train = np.asarray(range(self.n_samples_train))
            else:
                idx_train = resample(idx_X, n_samples=self.n_samples_train, replace=False, random_state=i, stratify=y)
            idx_test = np.asarray(list(set(idx_X) - set(idx_train)))
            yield idx_train, idx_test
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of split made by the splitter. 
        

        Parameters
        ----------
        X : ignored
            
        y : ignored
        
        groups : ignored
            

        Returns
        -------
        n_splits : int
            The n_splits attribute of the object.

        """
        return self.n_splits

class UCR_stratified_resample:
    """
    Class used as a splitter for sklearn cross validation tools. 
    It will take previsouly resampled arff files at a location and
    return a resample based on the identifier of the current cross
    validation step. 
    
    It is used to reproduce the exact same splits made in the original UCR/UEA
    archive. The arff files can be produced using the tsml java implementation.
    
    Parameters
    ----------
    n_splits : int
        Number of cross validation step planed.
    path : string
        Path to the arff files.
    
    """
    def __init__(self, n_splits, path):
        self.n_splits=n_splits
        self.path=path
    
    def split(self, X, y=None, groups=None):
        """
        

        Parameters
        ----------
        X : array, shape=(n_samples, n_features, n_timestamps)
            Time series data to split
        y : ignored
            
        groups : ignored
            

        Yields
        ------
        idx_Train : array, shape(n_samples_train)
            Index of the training data in the original dataset X.
        idx_Test : array, shape(n_samples_test)
            Index of the testing data in the original dataset X.

        """
        for i in range(self.n_splits):
            X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(self.path, i)
            idx_Train = [np.where((X == X_train[j]).all(axis=2))[0][0] for j in range(X_train.shape[0])]
            idx_Test = [np.where((X == X_test[j]).all(axis=2))[0][0] for j in range(X_test.shape[0])]
            yield idx_Train, idx_Test
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of split made by the splitter. 
        

        Parameters
        ----------
        X : ignored
            
        y : ignored
        
        groups : ignored
            

        Returns
        -------
        n_splits : int
            The n_splits attribute of the object.

        """
        return self.n_splits


def run_pipeline(pipeline, X_train, X_test, y_train, y_test, splitter, n_jobs=1):
    """
    Run a sklearn compatible model or pipeline on the specified dataset.

    Parameters
    ----------
    pipeline : object
        A sklearn compatible model or pipeline
    X_train : array, shape=(n_samples, n_feature, n_timestamps)
        Input training data
    X_test : array, shape=(n_samples, n_feature, n_timestamps)
        Input testing data
    y_train : array, shape=(n_samples)
        Class of the input training data.
    y_test : array, shape=(n_samples)
        Class of the input testing data.
    splitter : object
        A sklearn compatible splitter for cross-validation.
    n_jobs : int, optional
        Number of parallel validation round. The default is 1.

    Raises
    ------
    ValueError
        If the number of split of the splitter is an invalid value, raise a
        ValueError exception.

    Returns
    -------
    float
        Mean accuracy over all validation splits
    float
        Std of accuracy over all validation splits
    float
        Mean F1-score over all validation splits
    float
        Std F1-score over all validation splits
    float
        Mean runtime over all validation splits
    float
        Std runtime over all validation splits

    """
    if splitter.n_splits > 1:
        X = np.concatenate([X_train, X_test], axis=0).astype(np.float64)
        y = np.concatenate([y_train, y_test], axis=0).astype(np.float64)
        cv = cross_validate(pipeline, X, y, cv=splitter, n_jobs=n_jobs,
                            scoring={'f1': make_scorer(f1_score, average='macro'),
                                     'acc':make_scorer(accuracy_score)})
        return np.mean(cv['test_acc']), np.std(cv['test_acc']), np.mean(cv['test_f1']), np.std(cv['test_f1']), np.mean(cv['fit_time'] + cv['score_time']), np.std(cv['fit_time'] + cv['score_time'])

    if splitter.n_splits == 1:
        #change datetime to context accurate timing
        t0 = timer()
        pipeline = pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        t1 = timer()
        return  accuracy_score(y_test, pred), 0, f1_score(y_test, pred, average='macro'), 0, (t1-t0).total_seconds(), 0
    
    raise ValueError("Invalid value for n_split in splitter,"
                     " got {} splits".format(splitter.n_splits))

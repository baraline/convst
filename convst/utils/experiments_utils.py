# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:16:32 2022

@author: a694772
"""
import numpy as np
import pandas as pd
import copy

from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.utils import check_random_state

from timeit import default_timer as timer

from convst.utils.dataset_utils import load_UCR_UEA_dataset_split

#Reuse of the sktime function, modified for numpy array inputs rather than dataframes
# and to handle variable length series
def _resample(X_train, y_train, X_test, y_test, random_state):
    """Stratified resample data without replacement using a random state.

    Reproducable resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : np.array
        train data.
    y_train : np.array
        train data class labels.
    X_test : np.array
        test data.
    y_test : np.array
        test data class labes as np array.
    random_state : int
        seed to enable reproducable resamples
    Returns
    -------
    new train and test attributes and class labels.
    """
    all_labels = np.concatenate((y_train, y_test), axis=None)
    if isinstance(X_train, np.ndarray):
        all_data = np.concatenate([X_train, X_test],axis=0)
    else:
        all_data = X_train + X_test
    random_state = check_random_state(random_state)
    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    assert list(unique_train) == list(
        unique_test
    )  # haven't built functionality to deal with classes that exist in
    # test but not in train
    # prepare outputs
    if isinstance(X_train, np.ndarray):
        X_train = np.empty((0,X_train.shape[1],X_train.shape[2]))
        X_test = np.empty((0,X_test.shape[1],X_test.shape[2]))
    else:
        X_train = []
        X_test = []
    y_train = np.array([])
    y_test = np.array([])
    # for each class
    for label_index in range(0, len(unique_train)):
        # derive how many instances of this class from the counts
        num_instances = counts_train[label_index]
        # get the indices of all instances with this class label
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        # shuffle them
        random_state.shuffle(indices)
        # take the first lot of instances for train, remainder for test
        train_indices = indices[0:num_instances]
        test_indices = indices[num_instances:]
        del indices  # just to make sure it's not used!
        # extract data from corresponding indices
        # concat onto current data from previous loop iterations
        if isinstance(X_train, np.ndarray):
            train_instances = all_data[train_indices, :]
            test_instances = all_data[test_indices, :]
            X_train = np.concatenate([X_train, train_instances],axis=0)
            X_test = np.concatenate([X_test, test_instances],axis=0)
        else:
            for idx in train_indices:
                X_train.append(all_data[idx])
            for idx in test_indices:
                X_test.append(all_data[idx])
        train_labels = all_labels[train_indices]
        test_labels = all_labels[test_indices]
        y_train = np.concatenate([y_train, train_labels], axis=None)
        y_test = np.concatenate([y_test, test_labels], axis=None)
    # get the counts of the new train and test resample
    unique_train_new, counts_train_new = np.unique(y_train, return_counts=True)
    unique_test_new, counts_test_new = np.unique(y_test, return_counts=True)
    # make sure they match the original distribution of data
    assert list(counts_train_new) == list(counts_train)
    assert list(counts_test_new) == list(counts_test)
    return X_train, y_train, X_test, y_test


class cross_validate_UCR_UEA:
    def __init__(self, n_split, dataset_name, scorers={"accuracy":accuracy_score}):
        self.n_split = n_split
        self.dataset_name = dataset_name
        self.scorers = scorers
        
    def score(self, pipeline):
        X_train_0, X_test_0, y_train_0, y_test_0, _ = load_UCR_UEA_dataset_split(
            self.dataset_name
        )
        _score = pd.DataFrame()
        for i in range(self.n_split):
            
            if i == 0:
                if isinstance(X_train_0, np.ndarray):
                    X_train = np.copy(X_train_0)
                    X_test = np.copy(X_test_0)
                    
                else:
                    X_train = copy.copy(X_train_0)
                    X_test = copy.copy(X_test_0)
                y_train = np.copy(y_train_0)
                y_test = np.copy(y_test_0)
            else:
                X_train, y_train, X_test, y_test = _resample(
                    X_train_0, y_train_0, X_test_0, y_test_0, i
                )
            t0 = timer()
            pipeline = pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            t1 = timer()
            for scorer_name, scorer in self.scorers.items():
                _score.loc[i, scorer_name] = scorer(y_test, y_pred)
            _score.loc[i, 'time'] = t1 - t0
        return _score
        
    
class _sklearn_cv:
    def __init__(self, n_splits, dataset_name):
        self.n_splits=n_splits
        self.dataset_name=dataset_name
    
    def split(self, X, y=None, groups=None):
        X_train_0, X_test_0, y_train_0, y_test_0, _ = load_UCR_UEA_dataset_split(
            self.dataset_name
        )
        for i in range(self.n_splits):
            if i == 0:
                X_train = np.copy(X_train_0)
                X_test = np.copy(X_test_0)
                y_train = np.copy(y_train_0)
                y_test = np.copy(y_test_0)
            else:
                X_train, y_train, X_test, y_test = _resample(
                    X_train_0, y_train_0, X_test_0, y_test_0, i
                )
            idx_Train = [np.where((X == X_train[j]).all(axis=2))[0][0] for j in range(X_train.shape[0])]
            idx_Test = [np.where((X == X_test[j]).all(axis=2))[0][0] for j in range(X_test.shape[0])]
            yield idx_Train, idx_Test
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
class grid_search_UCR_UEA:
    def __init__(self, n_split, dataset_name, n_jobs, scorers=make_scorer(accuracy_score)):
        self.n_split = n_split
        self.dataset_name = dataset_name
        self.scorers = scorers
        self.n_jobs=n_jobs
        
    def score(self, pipeline, params):
        cv = GridSearchCV(
            pipeline(), params, scoring=self.scorers, n_jobs=self.n_jobs,
            cv=_sklearn_cv(self.n_split, self.dataset_name), pre_dispatch='n_jobs',
            verbose=3
        )
        X_train_0, X_test_0, y_train_0, y_test_0, _ = load_UCR_UEA_dataset_split(
            self.dataset_name
        )  
        X = np.concatenate((X_train_0, X_test_0),axis=0)
        y = np.concatenate((y_train_0, y_test_0),axis=0)
        cv.fit(X, y)
        return pd.DataFrame(cv.cv_results_)
    

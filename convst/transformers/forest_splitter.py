# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 21:23:20 2021

@author: Antoine
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import _tree
import numpy as np

class ForestSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, forest, leaves_only=True):
        self.leaves_only = leaves_only
        self.forest = forest
        self.n_trees = forest.get_params()['n_estimators']
        self.out_shape = None
        self.trees_id_split = None
        
    def fit(self, X, y=None):
        if self.leaves_only:
            #We only consider parent nodes of leaves
            n_splits = np.asarray([self.forest.estimators_[tree_id].tree_.n_leaves 
                                   for tree_id in range(self.n_trees)])
        else:
            #We consider all nodes
            n_nodes = np.asarray([self.forest.estimators_[i_dt].tree_.node_count
                                  for i_dt in range(self.n_trees)])
            n_leaves = np.asarray([self.forest.estimators_[i_dt].tree_.n_leaves
                                   for i_dt in range(self.n_trees)])
            n_splits = (n_nodes - n_leaves) * 2
        
        self.out_shape = (n_splits.sum(), X.shape[0])
        self.trees_id_split = np.zeros(self.n_trees+1, dtype=np.int32)
        self.trees_id_split[1:] += n_splits.cumsum()
        return self
        
    def transform(self, X, y=None):
        """
        Extract all data splits of the forest, with a split being, with 
        leaves_only to true, a tree node which has at least one leaf. Without, 
        all tree nodes are extracted.

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            Input data
        y : ignored

        Returns
        -------
        id_X_split : array, shape=(n_splits, n_samples)
            Indicate for each split of the tree which samples were used.
            The encoding is -1 for a non-used sample,
            0 for a sample that ended in the leaf, 1 for the other samples.
        ft_id : array, shape=(n_splits)
            Indicate the feature which was used in each split.

        """
        id_X_split =np.zeros(self.out_shape, dtype=np.int8) - 1
        ft_id = np.zeros(self.out_shape[0], dtype=np.int32)
        
        for tree_id in range(self.n_trees):
            tree = self.forest.estimators_[tree_id]
            node_indicator = tree.decision_path(X)
            if self.leaves_only:
                id_splits = np.where(tree.tree_.feature == _tree.TREE_UNDEFINED)[0]
            else:
                id_splits = np.where(tree.tree_.feature != _tree.TREE_UNDEFINED)[0]
            for i, id_split in enumerate(id_splits):
                if self.leaves_only:
                    #Get Parent node id
                    node_id = np.where((tree.tree_.children_right == id_split) | (
                        tree.tree_.children_left == id_split))[0]
                    if node_id.shape[0] > 0:
                        #Set samples in parent node to 1
                        id_X_split[self.trees_id_split[tree_id]+i,
                                    node_indicator[:, node_id[0]].nonzero()[0]] += 2
                        #Set samples in leaf node to 0
                        id_X_split[self.trees_id_split[tree_id]+i,
                                    node_indicator[:, id_split].nonzero()[0]] -= 1
                        #Select feature used in parent node
                        ft_id[self.trees_id_split[tree_id] +
                                  i] += tree.tree_.feature[node_id[0]]
                else:
                    #Here id_split is the parent node
                    id_right = tree.tree_.children_right[id_split]
                    id_left = tree.tree_.children_left[id_split]
                    
                    id_X_split[self.trees_id_split[tree_id]+i,
                                node_indicator[:, id_split].nonzero()[0]] += 2
                    id_X_split[self.trees_id_split[tree_id]+i,
                                node_indicator[:, id_right].nonzero()[0]] -= 1
                    ft_id[self.trees_id_split[tree_id] +
                              i] += tree.tree_.feature[id_split]

                    id_X_split[self.trees_id_split[tree_id]+i+len(id_splits),
                                node_indicator[:, id_split].nonzero()[0]] += 2 
                    id_X_split[self.trees_id_split[tree_id]+i+len(id_splits),
                                node_indicator[:, id_left].nonzero()[0]] -= 1
                    ft_id[self.trees_id_split[tree_id] +
                              i+len(id_splits)] += tree.tree_.feature[id_split]
        
        return id_X_split, ft_id
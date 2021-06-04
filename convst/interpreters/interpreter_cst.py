# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:58:02 2021

@author: A694772
"""
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.tree import _tree
from scipy.spatial.distance import cdist
from convst.utils import generate_strides_1D


class CST_interpreter():

    def __init__(self, cst, X_train, X_train_cst, y_train):
        self.cst = cst
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_cst = X_train_cst
        self.n_classes = np.unique(y_train).shape[0]

    def interpret_sample(self, X, model="DecisionTree", sns_context='talk'):
        sns.set(context=sns_context)
        if model == "DecisionTree":
            self._interpret_dt(X)
        elif model == "RidgeClassifier":
            self._interpret_ridge(X)
        else:
            raise ValueError("model parameter should be a string with as value "
                             "either DecisionTree or RidgeClassifier")

    def _interpret_dt(self, X):
        dt = DecisionTreeClassifier().fit(self.X_train_cst, self.y_train)
        tree = dt.tree_
        n_nodes = tree.node_count

        X_split = np.zeros(
            (n_nodes, self.X_train_cst.shape[0]), dtype=np.bool_)
        y_split = np.zeros(
            (n_nodes, self.X_train_cst.shape[0]), dtype=np.int8) - 1

        node_indicator = tree.decision_path(self.X_train_cst)
        for i in range(n_nodes):
            if tree.feature[i] != _tree.TREE_UNDEFINED:
                x_index_node = node_indicator[:, i].nonzero()[0]
                X_split[i, x_index_node] = True
                y_split[i, x_index_node] += (self.X_train_cst[
                    x_index_node, tree.feature[i]] <= tree.threshold[i]) + 1

        X_cst = self.cst.transform(X)
        node_indicator = dt.decision_path(X_cst)
        node_index = node_indicator.indices[node_indicator.indptr[0]:
                                            node_indicator.indptr[0 + 1]]

        df = pd.DataFrame(1/self.X_train_cst)
        df.columns = df.columns.astype(str)
        df['y'] = self.y_train
        pal = {k: sns.color_palette(n_colors=self.n_classes)[
            k] for k in range(self.n_classes)}
        for node_id in node_index:
            if tree.feature[node_id] == _tree.TREE_UNDEFINED:
                continue
            else:
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                df_node = df.loc[X_split[node_id], [
                    str(tree.feature[node_id]), "y"]]
                df_long = pd.melt(df_node, "y", var_name=" ", value_name="")
                ax[0].axhline(1/X_cst[0, tree.feature[node_id]],
                              color='red', linestyle='--')
                ax[0].axhline(1/tree.threshold[node_id],
                              color='black', linestyle='-.')
                sns.boxplot(x=" ", hue="y", y="",
                            palette=pal,
                            data=df_long, ax=ax[0], linewidth=2.5)

                ax[0].set_title("BoxPlot of 1/d for training samples")
                ax[0].set_xlabel(
                    "Shapelet n° {}".format(tree.feature[node_id]))
                ax[1].set_title("Shapelet")
                ax[2].set_title("Test sample with scaled shapelet")
                ax[2].plot(X[0, 0])
                dil = self.cst.dilation[tree.feature[node_id]]
                shp = self.cst.shapelets[tree.feature[node_id]]
                x = generate_strides_1D(X[0, 0], 9, dil)
                x = (x - x.mean(axis=-1, keepdims=True)) / \
                    x.std(axis=-1, keepdims=True)
                d = cdist(x, shp.reshape(1, 9), metric='sqeuclidean')
                loc = d.argmin()
                x_indexes = [loc + j*dil for j in range(9)]
                shp_v = (shp * X[0, 0, x_indexes].std()
                         ) + X[0, 0, x_indexes].mean()
                ax[2].scatter(x_indexes, shp_v, color='red')
                ax[1].scatter([0, 1, 2, 3, 4, 5, 6, 7, 8], shp, color='red')
                ax[1].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
                ax[1].set_xticklabels([j*dil for j in range(9)])
                plt.show()

    def _interpret_ridge(self, X):
        raise NotImplementedError()
        rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                                normalize=True, class_weight='balanced')
        rdg.fit(self.X_train_cst, self.y_train)
        coefs = rdg.coef_

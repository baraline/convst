# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:17:07 2021

@author: a694772
"""
#Totally Random Shapelet vs Rocket for two baselines
# Compare Performance vs Number of SHapelets
# Seuil a determiné par augmentation progressive du seuil,
# regarder combien d'occurence entre deux séries, et faire abs diff et prendre agrmax 

import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def class_vis(rdg, i_class, c):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(23,5))
    ix = np.zeros(rdg.coef_[0].shape[0])
    ix[1::3] = 1
    ix[2::3] = 2
    
    sns.boxplot(x=ix, y=rdg.coef_[i_class],ax=ax[0])
    ax[0].set_xticks([0,1,2])
    ax[0].set_xticklabels(['min','argmin','#match'])
    ax[0].set_ylabel('Ridge coef for class '+str(i_class))
    ax[1].set_ylabel('Ridge coef for class '+str(i_class))
    ax[1].set_xlabel('dilation')
    coef_sums = (rdg.coef_[i_class][0::3] + rdg.coef_[i_class][1::3] + rdg.coef_[i_class][2::3])/3
    sns.boxplot(x=c.dilation_, y=coef_sums,ax=ax[1])
    
    #sns.boxplot(x=c.length_, y=coef_sums,ax=ax[2])
    
def report(shp, ridge, X, y, xt=None, k=1, only_global=True):
    coefs = ridge.coef_
    if np.bincount(y).shape[0] == 2:
        coefs = np.append(-coefs, coefs, axis=0)
    
    for i_class in range(coefs.shape[0]):
        c = coefs[i_class]
        shp_coefs = c[1::3] + c[2::3] + c[0::3]
        idx = c.argsort()
        coefs_min = c[0::3]
        coefs_argmin = c[1::3]
        coefs_match = c[2::3]
        fig, ax = plt.subplots(ncols=2,figsize=(15,5))
        sns.distplot(coefs_min,label='min',ax=ax[0])
        sns.distplot(coefs_argmin,label='argmin',ax=ax[0])
        sns.distplot(coefs_match,label='match',ax=ax[0])
        ax[0].legend()
        ax[0].set_title('feature importance class {}'.format(i_class))
        sns.distplot(shp_coefs,ax=ax[1])
        plt.show()
        if not only_global:
            top_k = idx[-k:]//3
            low_k = idx[:k]//3
            for i_k in range(k):
                shp.visualise_one_shapelet(top_k[i_k], X, y, title='Top {} shapelet of class {}'.format(k, i_class))
                shp.visualise_one_shapelet(low_k[i_k], X, y, title='Worse {} shapelet of class {}'.format(k, i_class))
                if xt is not None:
                    fig, ax = plt.subplots(ncols=2, figsize=(15,5))
                    for j_class in range(coefs.shape[0]):
                        ax[0].scatter(xt[y==j_class, (top_k[i_k]*3)+0]*c[(top_k[i_k]*3)+0],
                                      xt[y==j_class, (top_k[i_k]*3)+2]*c[(top_k[i_k]*3)+2],
                                      c='C'+str(j_class),
                                      s=45, alpha=0.85)
                        ax[1].scatter(xt[y==j_class, (low_k[i_k]*3)+0]*c[(low_k[i_k]*3)+0],
                                      xt[y==j_class, (low_k[i_k]*3)+2]*c[(low_k[i_k]*3)+2],
                                      c='C'+str(j_class),
                                      s=45, alpha=0.85, label=j_class)
                        ax[1].legend()
                plt.show()
report(c, rf['ridgeclassifiercv'], X_test, y_test, rf['standardscaler'].transform(xt))
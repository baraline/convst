# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:25:16 2022

@author: antoi
"""

# TODO: Define class taking as input a RDST transformer,
# giving option to visualize each shapelets as base class

# Then, define interpreter for RDST Ridge

# And for RDST Ensemble, using RDST Ridge interpreter.


import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils.validation import check_is_fitted

from convst.transformers._commons import compute_shapelet_dist_vector, manhattan
from convst.transformers import R_DST
from convst.classifiers import R_DST_Ridge, R_DST_Ensemble

class Shapelet:
    def __init__(self, values, length, dilation, norm, threshold, phase, id=None):
        self.values = np.asarray(values)
        self.length = length
        self.dilation = dilation
        self.norm = norm
        self.phase = phase
        self.threshold = threshold
        self.id = id
        
    def plot(self, figsize=(10,5),  seaborn_context='talk'):
        sns.set()
        sns.set_context(seaborn_context)
        fig = plt.figure(figsize=(figsize))
        plt.plot(self.values)
        plt.xticks(np.arange(self.length), np.arange(self.length)*self.dilation)
        if self.id is not None:
            plt.title(
                'Shapelet n°{} (d={},normalize={},threshold={})'.format(
                    self.id, self.dilation, self.norm, self.threshold
                )
            )
        else:
            plt.title(
                'Shapelet (d={},normalize={},threshold={})'.format(
                    self.dilation, self.norm, self.threshold
                )
            )
        return fig
        
    def plot_on_X(
        self, X, d_func=manhattan, figsize=(10,5), seaborn_context='talk',
        shp_dot_size=30, shp_c='Orange'
    ):
        c = compute_shapelet_dist_vector(
            X, self.values, self.length, self.dilation,
            manhattan, self.norm, self.phase
        )
        sns.set()
        sns.set_context(seaborn_context)
        fig = plt.figure(figsize=(figsize))
        plt.plot(X)
        _values = self.values
        idx_match = np.asarray(
            [c.argmin() + i*self.dilation for i in range(self.length)]
        ).astype(int)
        if self.norm:
            _values = (_values * X[idx_match].std()) + X[idx_match].mean()
        plt.scatter(idx_match, _values, s=shp_dot_size, c=shp_c)
        
        return fig
    
    def plot_distance_vector(   
        self, X, d_func=manhattan, figsize=(10,5), seaborn_context='talk',
        c_threshold='Orange'
    ):
        c = compute_shapelet_dist_vector(
            X, self.values, self.length, self.dilation,
            manhattan, self.norm, self.phase
        )
        sns.set()
        sns.set_context(seaborn_context)
        fig = plt.figure(figsize=(figsize))
        plt.plot(c)
        plt.hlines(self.threshold, 0, c.shape[0], color=c_threshold)
        return fig

    
class RDST_interpreter():
    
    def __init__(self, RDST):
        check_is_fitted(RDST, ['shapelets_'])
        if isinstance(RDST, R_DST):
            self.RDST = RDST
        else:
            raise TypeError(
                'Object passed to RDST interpreter should be an R_DST instance'
            )
        
    def get_params(self, id_shapelet):
        values, lengths, dilations, threshold, normalize = self.RDST.shapelets_
        phase = self.RDST.phase_invariance
        if self.RDST.transform_type in ['multivariate','multivariate_variable']:
            raise NotImplementedError('Interpreter is not yet implemented for multivariate data')
        else:
            l = lengths[id_shapelet]
            v = values[id_shapelet, :l]
            d = dilations[id_shapelet]
            n = normalize[id_shapelet]
            t = threshold[id_shapelet]
        return v, l, d, n, t, phase 
        
        
    def plot_on_X(
        self, id_shapelet, X, d_func=manhattan, figsize=(10,5),
        seaborn_context='talk', shp_dot_size=30, shp_c='Orange'
    ):
        values, length, dilation, norm, threshold, phase = self.get_params(id_shapelet)
        return Shapelet(
            values, length, dilation, norm, threshold, phase
        ).plot_on_X(
            X, d_func=d_func, figsize=figsize,
            seaborn_context=seaborn_context, shp_dot_size=shp_dot_size,
            shp_c=shp_c
        )

    def plot_distance_vector(
        self, id_shapelet, X, d_func=manhattan, figsize=(10,5), 
        seaborn_context='talk', c_threshold='Orange'
    ):
        values, length, dilation, norm, threshold, phase = self.get_params(id_shapelet)
        return Shapelet(
            values, length, dilation, norm, threshold, phase
        ).plot_distance_vector()
    
    def plot(self, id_shapelet, figsize=(10,5), seaborn_context='talk'):
        values, length, dilation, norm, threshold, phase = self.get_params(id_shapelet)
        return Shapelet(
            values, length, dilation, norm, threshold, phase
        ).plot(figsize=figsize, seaborn_context=seaborn_context)
    

class RDST_Ridge_interpreter():
    def __init__(self, RDST_Ridge):
        
        check_is_fitted(RDST_Ridge, ['classifier'])
        if isinstance(RDST_Ridge, R_DST_Ridge):
            self.RDST_Ridge = RDST_Ridge
        else:
            raise TypeError(
                'Object passed to RDST_Ridge interpreter should be an R_DST_Ridge instance'
            )
    
    def get_shp_importance(self):
        """
        coefs = self.RDST_Ridge.classifier.coef_
        if coefs.shape[0] == 1:
            coefs = np.append(-coefs, coefs, axis=0)
        
        for i_class in range(coefs.shape[0]):
            c = coefs[i_class]
            shp_coefs = c[0::3] + c[1::3] + c[2::3] 
        """
        raise NotImplementedError()
        
        
class RDST_Ensemble_interpreter():
    def __init__(self, RDST_Ensemble):
        check_is_fitted(RDST_Ensemble, ['classifier'])
        if isinstance(RDST_Ensemble, R_DST_Ensemble):
            self.RDST_Ensemble = RDST_Ensemble
        else:
            raise TypeError(
                'Object passed to RDST_Ridge interpreter should be an R_DST_Ridge instance'
            )



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

#report(c, rf['ridgeclassifiercv'], X_test, y_test, rf['standardscaler'].transform(xt))
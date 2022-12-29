# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:25:16 2022
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
    def __init__(self, values, length, dilation, norm, threshold, phase):
        self.values = np.asarray(values)
        self.length = length
        self.dilation = dilation
        self.norm = norm
        self.phase = phase
        self.threshold = threshold
        
    def plot(self, figsize=(10,5), seaborn_context='talk', ax=None):
        if ax is None:
            sns.set()
            sns.set_context(seaborn_context)
            fig = plt.figure(figsize=(figsize))
            plt.plot(self.values)
            plt.title(
                'd={},normalize={},threshold={}'.format(
                    self.dilation, self.norm, np.round(self.threshold,decimals=2)
                )
            )
            return fig
        else:
            ax.plot(self.values)
            ax.set_title(
                'd={},normalize={},threshold={}'.format(
                    self.dilation, self.norm, np.round(self.threshold,decimals=2)
                )
            )
            return ax
        
    def plot_on_X(
        self, X, d_func=manhattan, figsize=(10,5), seaborn_context='talk', alpha=0.9,
        shp_dot_size=40, shp_c='purple', ax=None, label=None, x_linewidth=2
    ):
        c = compute_shapelet_dist_vector(
            X, self.values, self.length, self.dilation,
            manhattan, self.norm, self.phase
        )
        _values = self.values
        idx_match = np.asarray(
            [(c.argmin() + i*self.dilation)%X.shape[0] for i in range(self.length)]
        ).astype(int)
        if self.norm:
            _values = (_values * X[idx_match].std()) + X[idx_match].mean()
            
        if ax is None:
            sns.set()
            sns.set_context(seaborn_context)
            fig = plt.figure(figsize=(figsize))
            plt.plot(X,label=label, linewidth=x_linewidth, alpha=alpha)
            plt.scatter(idx_match, _values, s=shp_dot_size, c=shp_c, zorder=3, alpha=alpha)
            return fig
        else:
            ax.plot(X,label=label, linewidth=x_linewidth, alpha=alpha)
            ax.scatter(idx_match, _values, s=shp_dot_size, c=shp_c, zorder=3, alpha=alpha)
    
    def plot_distance_vector(   
        self, X, d_func=manhattan, figsize=(10,5), seaborn_context='talk',
        c_threshold='purple', ax=None, label=None
    ):
        c = compute_shapelet_dist_vector(
            X, self.values, self.length, self.dilation,
            manhattan, self.norm, self.phase
        )
        if ax is None:
            sns.set()
            sns.set_context(seaborn_context)
            fig = plt.figure(figsize=(figsize))
            plt.plot(c,label=label)
            plt.hlines(self.threshold, 0, c.shape[0], color=c_threshold)
            return fig
        else:
            ax.plot(c,label=label)
            ax.hlines(self.threshold, 0, c.shape[0], color=c_threshold)
            return ax

    
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
            length_ = lengths[id_shapelet]
            values_ = values[id_shapelet, :length_]
            dilation = dilations[id_shapelet]
            norm = normalize[id_shapelet]
            threshold_ = threshold[id_shapelet]
        return values_, length_, dilation, norm, threshold_, phase 
        
        
    def plot_on_X(
        self, id_shapelet, X, d_func=manhattan, figsize=(10,5),
        seaborn_context='talk', shp_dot_size=40, shp_c='purple', ax=None,
        label=None,
    ):
        return Shapelet(*self.get_params(id_shapelet)).plot_on_X(
            X, d_func=d_func, figsize=figsize,
            seaborn_context=seaborn_context, shp_dot_size=shp_dot_size,
            shp_c=shp_c, ax=ax, label=label
        )

    def plot_distance_vector(
        self, id_shapelet, X, d_func=manhattan, figsize=(10,5), 
        seaborn_context='talk', c_threshold='purple', ax=None, label=None
    ):
        return Shapelet(*self.get_params(id_shapelet)).plot_distance_vector(
            X, d_func=d_func, figsize=figsize, seaborn_context=seaborn_context,
            c_threshold=c_threshold, ax=ax, label=label
        )
    
    def plot(self, id_shapelet, figsize=(10,5), seaborn_context='talk', ax=None):
        return Shapelet(*self.get_params(id_shapelet)).plot(
            figsize=figsize, seaborn_context=seaborn_context, ax=ax
        )
    

class RDST_Ridge_interpreter():
    def __init__(self, RDST_Ridge):
        
        check_is_fitted(RDST_Ridge, ['classifier'])
        if isinstance(RDST_Ridge, R_DST_Ridge):
            self.RDST_Ridge = RDST_Ridge
        else:
            raise TypeError(
                'Object passed to RDST_Ridge interpreter should be an R_DST_Ridge instance'
            )
        self.rdst_interp = RDST_interpreter(self.RDST_Ridge.transformer)
        
    def get_shp_importance(self, class_id):
        coefs = self.RDST_Ridge.classifier['ridgeclassifiercv'].coef_
        n_classes = coefs.shape[0]
        
        if n_classes == 1:
            coefs = np.append(-coefs, coefs, axis=0)
        c_ = np.zeros(self.RDST_Ridge.transformer.shapelets_[1].shape[0]*3)
        c_[self.RDST_Ridge.classifier['c_standardscaler'].usefull_atts] = coefs[class_id]
        return c_
    
    def visualize_best_shapelets_one_class(
        self, X, y, class_id, n_shp=1, figsize=(16,12), seaborn_context='talk',
    ):
        sns.set()
        sns.set_context(seaborn_context)
        
        coefs = self.get_shp_importance(class_id)
        
        idx = (coefs.argsort()//3)[::-1]
        shp_ids = []
        i=0
        while len(shp_ids)<n_shp and i < idx.shape[0]:
            if idx[i] not in shp_ids:
                shp_ids = shp_ids + [idx[i]]
            i+=1
            
        X_new = self.RDST_Ridge.transformer.transform(X)
        i_example = np.random.choice(np.where(y==class_id)[0])
        i_example2 = np.random.choice(np.where(y!=class_id)[0])
        y_copy = (y == class_id).astype(int)
        for i_shp in shp_ids:
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=figsize)
            
            ax[0,0].set_title('Boxplot of min')
            sns.boxplot(x=y_copy,y=X_new[:,(i_shp*3)],ax=ax[0,0])
            ax[0,0].set_xticklabels(['Other classes', 'Class {}'.format(class_id)])
            
            ax[0,1].set_title('Boxplot of argmin')
            sns.boxplot(x=y_copy,y=X_new[:,1+(i_shp*3)],ax=ax[0,1])
            
            ax[0,1].set_xticklabels(['Other classes', 'Class {}'.format(class_id)])
            ax[0,2].set_title('Boxplot of Shapelet Occurence')
            sns.boxplot(x=y_copy,y=X_new[:,2+(i_shp*3)],ax=ax[0,2])
            
            ax[0,2].set_xticklabels(['Other classes', 'Class {}'.format(class_id)])            
            
            ax[1,0].set_title('Best match')
            ax[1,2].set_title('Distance vectors')
            
            
            self.rdst_interp.plot(i_shp, ax=ax[1,1])
            
            self.rdst_interp.plot_on_X(i_shp, X[i_example2,0], ax=ax[1,0], label='Other class')
            self.rdst_interp.plot_on_X(i_shp, X[i_example,0], ax=ax[1,0], label='Class {}'.format(class_id))
            
            self.rdst_interp.plot_distance_vector(i_shp, X[i_example2,0], ax=ax[1,2], label='Other class')
            self.rdst_interp.plot_distance_vector(i_shp, X[i_example,0], ax=ax[1,2], label='Class {}'.format(class_id))
        
            ax[1,0].legend()

            
class RDST_Ensemble_interpreter:
    def __init__(self, RDST_Ensemble):
        check_is_fitted(RDST_Ensemble, ['classifier'])
        if isinstance(RDST_Ensemble, R_DST_Ensemble):
            self.RDST_Ensemble = RDST_Ensemble
        else:
            raise TypeError(
                'Object passed to RDST_Ridge interpreter should be an R_DST_Ridge instance'
            )
    
    def visualize_best_shapelets_one_class(self, class_id, n_shp=1):
        raise NotImplementedError()
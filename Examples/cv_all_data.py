# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:21:00 2021

@author: A694772
"""
from sktime.transformations.panel.rocket import MiniRocket as MiniRKT
from CST.utils.dataset_utils import load_sktime_dataset, return_all_dataset_names
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import numpy as np
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from pyts.classification import LearningShapelets
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from datetime import datetime

resume=False

print("Imports OK")
n_cv = 10
n_splits=10
p=[100,95,90,85,80]

available_memory_bytes = 62*1e9
max_cpu_cores = 86
numba_n_thread = 3
size_mult = 3500

max_process = max_cpu_cores//numba_n_thread

csv_name = 'CV_{}_results_{}_{}.csv'.format(n_cv,n_splits,p)

dataset_names = return_all_dataset_names()    

if resume:
    df = pd.read_csv(csv_name)
else:
    df = pd.DataFrame(index=dataset_names)
    df['MiniCST_mean'] = pd.Series(0, index=df.index)
    df['MiniCST_std'] = pd.Series(0, index=df.index)
    df['MiniRKT_mean'] = pd.Series(0, index=df.index)
    df['MiniRKT_std'] = pd.Series(0, index=df.index)
    df['Runtime_MiniCST'] = pd.Series('0', index=df.index)
    df['Runtime_MiniRKT'] = pd.Series('0', index=df.index)
    df['MiniCST_n_shp'] = pd.Series(0, index=df.index)
    df['MiniCST_n_shp_used'] = pd.Series(0, index=df.index)
    df['MiniCST_n_kernel'] = pd.Series(0, index=df.index)


def n_shp_extracted(pipelines):
    return np.mean([pipeline['miniconvolutionalshapelettransformer'].n_shapelets 
                    for pipeline in pipelines])

def n_shp_used(pipelines):
    return np.mean([(pipeline['randomforestclassifier'].feature_importances_ > 0).sum()
                    for pipeline in pipelines])

def n_kernels(pipelines):
    return np.mean([pipeline['miniconvolutionalshapelettransformer'].n_kernels
                    for pipeline in pipelines])

for name in dataset_names:
    print(name)
    X, y, le = load_sktime_dataset(name,normalize=True)
    n_jobs = int(available_memory_bytes // (X.nbytes * size_mult))
    n_jobs = max(n_jobs if n_jobs <= max_process else max_process, 1)
    if df.loc[name,'MiniCST_mean'] == 0 and X.shape[2] > 10:
        
        pipe_rkt = make_pipeline(MiniRKT(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))
        cv = cross_validate(pipe_rkt, X, y, cv=n_cv, n_jobs=n_jobs,
                            scoring={'f1':make_scorer(f1_score, average='macro')})
        print("F1-Score for MINI-ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
        df.loc[name,'MiniRKT_mean'] = np.mean(cv['test_f1'])
        df.loc[name,'MiniRKT_std'] = np.std(cv['test_f1'])
        df.loc[name,'Runtime_MiniRKT'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        pipe_cst = make_pipeline(MiniConvolutionalShapeletTransformer(n_threads=numba_n_thread),
                                 RandomForestClassifier(n_estimators=400, max_samples=0.75))
        
        cv = cross_validate(pipe_cst, X, y, cv=n_cv, 
                            scoring={'f1':make_scorer(f1_score, average='macro')},
                            n_jobs=n_jobs, return_estimator=True)
        
        print("F1-Score for MiniCST RF : {}".format(np.mean(cv['test_f1'])))
        df.loc[name,'MiniCST_mean'] =  np.mean(cv['test_f1'])
        df.loc[name,'MiniCST_std'] = np.std(cv['test_f1'])
        df.loc[name,'Runtime_MiniCST'] = np.mean(cv['fit_time'] + cv['score_time'])
        df.loc[name,'MiniCST_n_shp'] = n_shp_extracted(cv['estimator'])
        df.loc[name,'MiniCST_n_shp_used'] = n_shp_extracted(cv['estimator'])
        df.loc[name,'MiniCST_n_kernel'] = n_kernels(cv['estimator'])
            
        df.to_csv(csv_name)
        print('---------------------')
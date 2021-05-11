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
from wildboar.ensemble import ShapeletForestClassifier

import warnings
#Can use this to resume to last dataset if a problem occured
resume = True

print("Imports OK")
n_cv = 10
n_splits = 10
p = [100, 95, 90, 85, 80]

run_RKT = True
run_CST = True
run_SFC = True

available_memory_bytes = 60 *1e9
max_cpu_cores = 30
numba_n_thread = 3
size_mult = 4000
random_state = None

max_process = max_cpu_cores//numba_n_thread

csv_name = 'CV_{}_results_{}_{}_final.csv'.format(n_cv, n_splits, p)

dataset_names = return_all_dataset_names()

if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
    df = df.drop(df.index[np.where(~df.index.isin(dataset_names))[0]],axis=0)
    df.to_csv(csv_name)
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
    df['SFC_mean'] = pd.Series(0,index=df.index)
    df['SFC_std'] = pd.Series(0,index=df.index)
    df['Runtime_SFC'] = pd.Series('0', index=df.index)
    df.to_csv(csv_name)


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
    X, y, le = load_sktime_dataset(name, normalize=True)
    n_possible_jobs = int(available_memory_bytes // (X.nbytes * size_mult))
    n_jobs = max(n_possible_jobs if n_possible_jobs <=
                 max_process else max_process, 1)
    if n_possible_jobs == 0:
        warnings.warn("Not enought estimated memory to run current dataset")
    else:
        if run_RKT and df.loc[name, 'MiniRKT_mean'] == 0 and X.shape[2] > 10:
            pipe_rkt = make_pipeline(MiniRKT(random_state=random_state),
                                     RidgeClassifierCV(alphas=np.logspace(-4, 4, 10), normalize=True))
            cv = cross_validate(pipe_rkt, X, y, cv=n_cv, n_jobs=n_jobs,
                                scoring={'f1': make_scorer(f1_score, average='macro')})
            print(
                "F1-Score for MINI-ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
            df.loc[name, 'MiniRKT_mean'] = np.mean(cv['test_f1'])
            df.loc[name, 'MiniRKT_std'] = np.std(cv['test_f1'])
            df.loc[name, 'Runtime_MiniRKT'] = np.mean(
                cv['fit_time'] + cv['score_time'])
            df.to_csv(csv_name)

        if run_CST and df.loc[name, 'MiniCST_mean'] == 0 and X.shape[2] > 10:
            pipe_cst = make_pipeline(MiniConvolutionalShapeletTransformer(n_threads=numba_n_thread,
                                                                          random_state=random_state),
                                     RandomForestClassifier(n_estimators=400,
                                                            random_state=random_state))
            cv = cross_validate(pipe_cst, X, y, cv=n_cv,
                                scoring={'f1': make_scorer(
                                    f1_score, average='macro')},
                                n_jobs=n_jobs, return_estimator=True)
            print(
                "F1-Score for MiniCST RF : {}".format(np.mean(cv['test_f1'])))
            df.loc[name, 'MiniCST_mean'] = np.mean(cv['test_f1'])
            df.loc[name, 'MiniCST_std'] = np.std(cv['test_f1'])
            df.loc[name, 'Runtime_MiniCST'] = np.mean(
                cv['fit_time'] + cv['score_time'])
            df.loc[name, 'MiniCST_n_shp'] = n_shp_extracted(cv['estimator'])
            df.loc[name, 'MiniCST_n_shp_used'] = n_shp_extracted(
                cv['estimator'])
            df.loc[name, 'MiniCST_n_kernel'] = n_kernels(cv['estimator'])

            df.to_csv(csv_name)
        
        if run_SFC and df.loc[name, 'SFC_mean'] == 0 and X.shape[2] > 10:
            cv = cross_validate(ShapeletForestClassifier(n_estimators=400, 
                                                         random_state=random_state),
                                X[:,0,:], y, cv=n_cv,
                                scoring={'f1': make_scorer(
                                    f1_score, average='macro')},
                                n_jobs=n_jobs)
            print(
                "F1-Score for SFC : {}".format(np.mean(cv['test_f1'])))
            df.loc[name, 'SFC_mean'] = np.mean(cv['test_f1'])
            df.loc[name, 'SFC_std'] = np.std(cv['test_f1'])
            df.loc[name, 'Runtime_SFC'] = np.mean(
                cv['fit_time'] + cv['score_time'])
            df.to_csv(csv_name)
        
    print('---------------------')

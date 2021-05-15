# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:21:00 2021

@author: A694772
"""
import pandas as pd
import numpy as np

from datetime import datetime
from sktime.transformations.panel.rocket import MiniRocket as MiniRKT
from CST.utils.dataset_utils import load_sktime_arff_file_resample_id, return_all_dataset_names, UCR_stratified_resample
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from wildboar.ensemble import ShapeletForestClassifier

import warnings
#Can use this to resume to last dataset if a problem occured
resume = False

print("Imports OK")
#n_cv = 1 to test on original train test split, more to make stratified k folds
n_cv = 30
max_samples = 0.2
P = 80
n_bins = 9

run_RKT = True
run_CST = False
run_SFC = True

available_memory_bytes = 60 * 1e9
max_cpu_cores = 90
numba_n_thread = 3
size_mult = 1500
random_state = None

max_process = max_cpu_cores//numba_n_thread

csv_name = 'CV_{}_results_{}_{}_{}.csv'.format(n_cv, max_samples, n_bins, P)
base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"

dataset_names = return_all_dataset_names()


if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
    df = df.drop(df.index[np.where(~df.index.isin(dataset_names))[0]],axis=0)
    df.to_csv(csv_name)
else:
    df = pd.DataFrame(index=dataset_names)
    df['CST_mean'] = pd.Series(0, index=df.index)
    df['CST_std'] = pd.Series(0, index=df.index)
    df['CST_runtime'] = pd.Series(0, index=df.index)
    df['MiniRKT_mean'] = pd.Series(0, index=df.index)
    df['MiniRKT_std'] = pd.Series(0, index=df.index)
    df['MiniRKT_runtime'] = pd.Series(0, index=df.index)
    df['SFC_mean'] = pd.Series(0,index=df.index)
    df['SFC_std'] = pd.Series(0,index=df.index)
    df['SFC_runtime'] = pd.Series(0, index=df.index)
    df.to_csv(csv_name)
    
        
def run_pipeline(pipeline, X_train, X_test, y_train, y_test, splitter, n_jobs):
    if splitter.n_splits > 1:
        X = np.concatenate([X_train, X_test],axis=0)
        y = np.concatenate([y_train, y_test],axis=0)
        cv = cross_validate(pipeline, X, y, cv=splitter, n_jobs=n_jobs,
                            scoring={'acc': make_scorer(f1_score, average='macro')})
        return np.mean(cv['test_acc']),  np.std(cv['test_acc']), np.mean(cv['fit_time'] + cv['score_time'])
    
    elif splitter.n_splits == 1:
        t0 = datetime.now()
        pipeline = pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        t1 = datetime.now()
        return [f1_score(y_test, pred, average='macro')], [0], [(t1-t0).total_seconds()]

pipe_rkt = make_pipeline(MiniRKT(random_state=random_state), 
                         RidgeClassifierCV(alphas=np.logspace(-4, 4, 10), 
                                           normalize=True))
pipe_cst = make_pipeline(ConvolutionalShapeletTransformer(n_threads=numba_n_thread,
                                                          P=P,
                                                          max_samples=max_samples,
                                                          n_bins=n_bins,
                                                          random_state=random_state),
                         RandomForestClassifier(n_estimators=400,
                                                random_state=random_state))

pipe_sfc = make_pipeline(ShapeletForestClassifier(random_state=random_state))

for name in dataset_names:
    print(name)
    ds_path = base_UCR_resamples_path+"{}/{}".format(name,name)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(ds_path, 0, normalize=True)
    n_possible_jobs = min(int(available_memory_bytes // ((X_train.nbytes + X_test.nbytes) * size_mult)), n_cv)
    n_jobs = max(n_possible_jobs if n_possible_jobs <=
                 max_process else max_process, 1)
    
    splitter = UCR_stratified_resample(n_cv, ds_path)
    
    if n_possible_jobs == 0:
        warnings.warn("Not enought estimated memory to run current dataset")
    else:
        if run_RKT and df.loc[name, 'MiniRKT_mean'] == 0:
            mean, std, time = run_pipeline(pipe_rkt, X_train, X_test, y_train, y_test, splitter, n_jobs)
            df.loc[name, 'MiniRKT_mean'] = mean
            df.loc[name, 'MiniRKT_std'] = std
            df.loc[name, 'MiniRKT_runtime'] = time
            df.to_csv(csv_name)

        if run_CST and df.loc[name, 'CST_mean'] == 0:
            mean, std, time = run_pipeline(pipe_cst, X_train, X_test, y_train, y_test, splitter, n_jobs)
            df.loc[name, 'CST_mean'] = mean
            df.loc[name, 'CST_std'] = std
            df.loc[name, 'CST_runtime'] = time
            df.to_csv(csv_name)
        
        if run_SFC and df.loc[name, 'SFC_mean'] == 0:
            mean, std, time = run_pipeline(pipe_sfc, X_train, X_test, y_train, y_test, splitter, n_jobs)
            df.loc[name, 'SFC_mean'] = mean
            df.loc[name, 'SFC_std'] = std
            df.loc[name, 'SFC_runtime'] = time
            df.to_csv(csv_name)
        
    print('---------------------')

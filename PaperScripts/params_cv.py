# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:30:45 2021

@author: A694772
"""
import pandas as pd
import numpy as np

from CST.utils.dataset_utils import load_sktime_arff_file_resample_id, return_all_dataset_names, UCR_stratified_resample
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


#Can use this to resume to last dataset if a problem occured
resume = False

available_memory_bytes = 60 * 1e9
max_cpu_cores = 96
numba_n_thread = 3
size_mult = 3500

max_process = max_cpu_cores//numba_n_thread
file_name = 'params_csv.csv'
n_cv = 10
base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"

P = [95,90,85,80]
max_samples = [0.1,0.15,0.2,0.25,0.3]
n_bins = [5, 7, 9, 11, 13, 15]
params = {'CST__P': P,
          'CST__max_samples': max_samples,
          'CST__n_bins':n_bins}
print(params)

if resume:
    df = pd.read_csv(file_name, sep=';')
    df = df.set_index('Unnamed: 0')
else:
    df = pd.DataFrame()

dataset_names = return_all_dataset_names()

for name in dataset_names:
    results = {}
    print(name)
    if name not in df.index.values:
        ds_path = base_UCR_resamples_path+"{}/{}".format(name,name)
        X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(ds_path, 0, normalize=True)
        n_possible_jobs = int(available_memory_bytes // ((X_train.nbytes + X_test.nbytes) * size_mult))
        n_jobs = max(n_possible_jobs if n_possible_jobs <=
                 max_process else max_process, 1)  
        print(n_jobs)
        if n_jobs >= 30:
            X = np.concatenate([X_train, X_test],axis=0)
            y = np.concatenate([y_train, y_test],axis=0)
            splitter = UCR_stratified_resample(n_cv, ds_path)
            pipe = Pipeline([('CST', ConvolutionalShapeletTransformer()),
                             ('rf', RandomForestClassifier(n_estimators=400))])
            clf = GridSearchCV(
                pipe, params, n_jobs=n_jobs, cv=splitter, verbose=1)
            clf.fit(X, y)
            
            p_key = clf.cv_results_['params']
            rank = clf.cv_results_['mean_test_score']

            for i, p in enumerate(p_key):
                if str(p) in results.keys():
                    results[str(p)].append(rank[i])
                else:
                    results.update({str(p): [rank[i]]})
            print('Dumping results to csv')
            df = pd.concat(
                [df, pd.DataFrame(results, index=[name])], axis=0)
            df.to_csv(file_name)

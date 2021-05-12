# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:30:45 2021

@author: A694772
"""
from CST.utils.dataset_utils import load_sktime_dataset, return_all_dataset_names
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from sklearn.pipeline import Pipeline
from itertools import combinations
from sklearn.model_selection import GridSearchCV


#Can use this to resume to last dataset if a problem occured
resume = False

available_memory_bytes = 62*1e9
max_cpu_cores = 86
numba_n_thread = 3
size_mult = 3500

max_process = max_cpu_cores//numba_n_thread
file_name = 'params_csv.csv'
n_cv = 10


ps = []
for r in range(1, 6):
    ps.extend(list(combinations([100, 95, 90, 85, 80], r)))
n_splits = [1, 3, 5, 7, 10]
n_bins = [5, 7, 9, 11, 13, 15, 17, 20]
params = {'CST__P': ps,
          'CST__n_splits': n_splits,
          'CST__n_bins':n_bins}
print(params)

if resume:
    df = pd.read_csv(file_name, sep=';')
    df = df.set_index('Unnamed: 0')
    print(df.index.values)
else:
    df = pd.DataFrame()

dataset_names = return_all_dataset_names()

for d_name in dataset_names:
    results = {}
    print(d_name)
    if d_name not in df.index.values:
        X, y, le = load_sktime_dataset(d_name, normalize=True)
        if X.shape[2] > 10:
            n_jobs = int(available_memory_bytes // (X.nbytes * size_mult))
            n_jobs = max(n_jobs if n_jobs <= max_process else max_process, 1)
            if n_jobs >= 15:
                print('Launching {} parallel jobs'.format(n_jobs))
                pipe = Pipeline([('CST', ConvolutionalShapeletTransformer()),
                                 ('rf', RandomForestClassifier(n_estimators=400, ccp_alpha=0.05))])
                clf = GridSearchCV(
                    pipe, params, n_jobs=n_jobs, cv=n_cv, verbose=1)
                clf.fit(X, y)
                print('Done')
                p_key = clf.cv_results_['params']
                rank = clf.cv_results_['mean_test_score']

                for i, p in enumerate(p_key):
                    if str(p) in results.keys():
                        results[str(p)].append(rank[i])
                    else:
                        results.update({str(p): [rank[i]]})
                print('Dumping results to csv')
                df = pd.concat(
                    [df, pd.DataFrame(results, index=[d_name])], axis=0)
                df.to_csv(file_name, sep=';')

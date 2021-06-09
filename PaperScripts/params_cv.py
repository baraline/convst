# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils import load_sktime_arff_file_resample_id, return_all_dataset_names, UCR_stratified_resample
from convst.transformers import ConvolutionalShapeletTransformer

from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# TODO : update script with newest API.

# Can use this to resume to last dataset if a problem occured
resume = False

available_memory_bytes = 60 * 1e9
max_cpu_cores = 32
tree_jobs = 4
size_mult = 3250


max_process = max_cpu_cores//tree_jobs
file_name = 'params_csv.csv'
n_cv = 10
base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"

P = [95, 90, 85, 80]
n_trees = [100,200,300,400]
n_bins = [7, 9, 11, 13]

params = {'CST__P': P,
          'CST__n_trees': n_trees,
          'CST__n_bins':n_bins}

print(params)

if resume:
    df = pd.read_csv(file_name)
    df = df.set_index('Unnamed: 0')
else:
    df = pd.DataFrame()

dataset_names = return_all_dataset_names()

# TODO: extract dataset name to put them in the loop for reproductibility

for name in dataset_names:
    results = {}
    print(name)
    if name not in df.index.values:
        ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
        X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
            ds_path, 0, normalize=True)
        n_possible_jobs = int(available_memory_bytes //
                              ((X_train.nbytes + X_test.nbytes) * size_mult))
        n_jobs = max(n_possible_jobs if n_possible_jobs <=
                     max_process else max_process, 1)
        print(n_jobs)
        if n_jobs >= 25:
            X = np.concatenate([X_train, X_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            splitter = UCR_stratified_resample(n_cv, ds_path)
            pipe = Pipeline([('CST', ConvolutionalShapeletTransformer(n_jobs=tree_jobs)),
                             ('rdg', RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))])
            clf = GridSearchCV(
                pipe, params, n_jobs=n_jobs//2, cv=splitter, verbose=1)
            clf.fit(X, y)
            print('Done!')
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

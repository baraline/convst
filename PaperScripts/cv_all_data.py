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
from CST.shapelet_transforms.try_CST import ConvolutionalShapeletTransformer_tree
from sktime.classification.shapelet_based import MrSEQLClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from wildboar.ensemble import ShapeletForestClassifier
from sklearn.metrics import accuracy_score

#Can use this to resume to last dataset if a problem occured
resume = False

print("Imports OK")
#n_cv = 1 to test on original train test split, more to make stratified k folds
n_cv = 30
n_trees=200
max_ft=1.0
P = 80
n_bins = 9

run_RKT = True
run_CST = True
run_MrSEQL = True
run_SFC = True

available_memory_bytes = 60 * 1e9
max_cpu_cores = 90
numba_n_thread = 3
size_mult = 2500
random_state = None

max_process = max_cpu_cores//numba_n_thread

csv_name = 'CV_{}_results_({},{})_{}_{}.csv'.format(
    n_cv, n_trees, max_ft, n_bins, P)
base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"

dataset_names = return_all_dataset_names()


if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
    df = df.drop(df.index[np.where(~df.index.isin(dataset_names))[0]], axis=0)
    df.to_csv(csv_name)
else:
    df = pd.DataFrame(index=dataset_names)
    df['CST_f1_mean'] = pd.Series(0, index=df.index)
    df['CST_f1_std'] = pd.Series(0, index=df.index)
    df['CST_acc_mean'] = pd.Series(0, index=df.index)
    df['CST_acc_std'] = pd.Series(0, index=df.index)
    df['CST_runtime'] = pd.Series(0, index=df.index)
    df['MiniRKT_f1_mean'] = pd.Series(0, index=df.index)
    df['MiniRKT_f1_std'] = pd.Series(0, index=df.index)
    df['MiniRKT_acc_mean'] = pd.Series(0, index=df.index)
    df['MiniRKT_acc_std'] = pd.Series(0, index=df.index)
    df['MiniRKT_runtime'] = pd.Series(0, index=df.index)
    df['MrSEQL_f1_mean'] = pd.Series(0, index=df.index)
    df['MrSEQL_f1_std'] = pd.Series(0, index=df.index)
    df['MrSEQL_acc_mean'] = pd.Series(0, index=df.index)
    df['MrSEQL_acc_std'] = pd.Series(0, index=df.index)
    df['MrSEQL_runtime'] = pd.Series(0, index=df.index)
    df['SFC_f1_mean'] = pd.Series(0, index=df.index)
    df['SFC_f1_std'] = pd.Series(0, index=df.index)
    df['SFC_acc_mean'] = pd.Series(0, index=df.index)
    df['SFC_acc_std'] = pd.Series(0, index=df.index)
    df['SFC_runtime'] = pd.Series(0, index=df.index)
    df.to_csv(csv_name)


def run_pipeline(pipeline, X_train, X_test, y_train, y_test, splitter, n_jobs):
    if splitter.n_splits > 1:
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        cv = cross_validate(pipeline, X, y, cv=splitter, n_jobs=n_jobs,
                            scoring={'f1': make_scorer(f1_score, average='macro'),
                                     'acc':make_scorer(accuracy_score)})
        return np.mean(cv['test_acc']),  np.std(cv['test_acc']), np.mean(cv['test_f1']),  np.std(cv['test_f1']), np.mean(cv['fit_time'] + cv['score_time'])

    elif splitter.n_splits == 1:
        t0 = datetime.now()
        pipeline = pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        t1 = datetime.now()
        return [f1_score(y_test, pred, average='macro'), accuracy_score(y_test, pred)], [0], [(t1-t0).total_seconds()]


pipe_rkt = make_pipeline(MiniRKT(random_state=random_state),
                         RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                                           normalize=True))

pipe_cst = make_pipeline(ConvolutionalShapeletTransformer_tree(n_threads=numba_n_thread,
                                                          P=P,
                                                          n_trees=n_trees,
                                                          max_ft=max_ft,
                                                          n_bins=n_bins,
                                                          random_state=random_state),
                         RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))

pipe_sfc = make_pipeline(ShapeletForestClassifier(random_state=random_state))

pipe_MrSEQL = make_pipeline(MrSEQLClassifier(symrep=['sax','sfa']))

for name in dataset_names:
    print(name)
    ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
        ds_path, 0, normalize=True)

    splitter = UCR_stratified_resample(n_cv, ds_path)

    if run_RKT and df.loc[name, 'MiniRKT_f1_mean'] == 0:
        n_possible_jobs = min(int(available_memory_bytes //
                      ((X_train.nbytes + X_test.nbytes) * 1000)), n_cv)
        n_jobs = max(n_possible_jobs if n_possible_jobs <=
                     max_process else max_process, 1)
        print("Processing RKT with {} jobs".format(n_jobs))
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_rkt, X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'MiniRKT_acc_mean'] = acc_mean
        df.loc[name, 'MiniRKT_acc_std'] = acc_std
        df.loc[name, 'MiniRKT_f1_mean'] = f1_mean
        df.loc[name, 'MiniRKT_f1_std'] = f1_std
        df.loc[name, 'MiniRKT_runtime'] = time
        df.to_csv(csv_name)

    if run_CST and df.loc[name, 'CST_f1_mean'] == 0:
        n_possible_jobs = min(int(available_memory_bytes //
              ((X_train.nbytes + X_test.nbytes) * size_mult)), n_cv)
        n_jobs = max(n_possible_jobs if n_possible_jobs <=
                     max_process else max_process, 1)
        print("Processing CST with {} jobs".format(n_jobs))
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_cst, X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'CST_acc_mean'] = acc_mean
        df.loc[name, 'CST_acc_std'] = acc_std
        df.loc[name, 'CST_f1_mean'] = f1_mean
        df.loc[name, 'CST_f1_std'] = f1_std
        df.loc[name, 'CST_runtime'] = time
        df.to_csv(csv_name)

    if run_MrSEQL and df.loc[name, 'MrSEQL_f1_mean'] == 0:
        n_possible_jobs = min(int(available_memory_bytes //
              ((X_train.nbytes + X_test.nbytes) * 1000)), n_cv)
        n_jobs = max(n_possible_jobs if n_possible_jobs <=
                     max_process else max_process, 1)
        print("Processing MrSEQL with {} jobs".format(n_jobs))
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_MrSEQL, X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'MrSEQL_acc_mean'] = acc_mean
        df.loc[name, 'MrSEQL_acc_std'] = acc_std
        df.loc[name, 'MrSEQL_f1_mean'] = f1_mean
        df.loc[name, 'MrSEQL_f1_std'] = f1_std
        df.loc[name, 'MrSEQL_runtime'] = time
        df.to_csv(csv_name)

    if run_SFC and df.loc[name, 'SFC_f1_mean'] == 0:
        n_possible_jobs = min(int(available_memory_bytes //
              ((X_train.nbytes + X_test.nbytes) * 1000)), n_cv)
        n_jobs = max(n_possible_jobs if n_possible_jobs <=
                     max_process else max_process, 1)
        print("Processing SFC with {} jobs".format(n_jobs))
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_sfc, X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'SFC_acc_mean'] = acc_mean
        df.loc[name, 'SFC_acc_std'] = acc_std
        df.loc[name, 'SFC_f1_mean'] = f1_mean
        df.loc[name, 'SFC_f1_std'] = f1_std
        df.loc[name, 'SFC_runtime'] = time
        df.to_csv(csv_name)

print('---------------------')
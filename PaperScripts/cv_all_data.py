# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from datetime import datetime

from sktime.transformations.panel.rocket import MiniRocket as MiniRKT
from sktime.classification.shapelet_based import MrSEQLClassifier

from convst.utils import load_sktime_arff_file_resample_id, return_all_dataset_names, UCR_stratified_resample
from convst.transformers import ConvolutionalShapeletTransformer_onlyleaves as ConvolutionalShapeletTransformer

from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import accuracy_score

from wildboar.ensemble import ShapeletForestClassifier

from numba import set_num_threads
#Can use this to resume to last dataset if a problem occured
resume = False

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Modify this to your path to the UCR resamples, check README for how to get them. 
# Another splitter is also provided in dataset_utils to make random resamples

base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print("Imports OK")
#n_cv = 1 to test on original train test split.
n_cv=30
n_trees=200
max_ft=1.0
P=80
n_bins=11
random_state = None

run_RKT = False
run_CST = True
run_MrSEQL = False
run_SFC = False

#Machine parameters, to change with yours.
available_memory_bytes = 60 * 1e9
n_cores = 90

def get_n_jobs_n_threads(nbytes, size_mult=3000):
    nbytes *= size_mult
    n_jobs = min(max(available_memory_bytes//nbytes,1),10)
    n_threads = max(n_cores//n_jobs,1)
    return int(n_jobs), int(n_threads)

csv_name = 'CV_only_leaves_{}_results_({},{})_{}_{}.csv'.format(
    n_cv, n_trees, max_ft, n_bins, P)

dataset_names = return_all_dataset_names()

if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
    df = df.drop(df.index[np.where(~df.index.isin(dataset_names))[0]], axis=0)
    df.to_csv(csv_name)
else:
    df = pd.DataFrame(index=dataset_names)
    if run_CST:
        df['CST_f1_mean'] = pd.Series(0, index=df.index)
        df['CST_f1_std'] = pd.Series(0, index=df.index)
        df['CST_acc_mean'] = pd.Series(0, index=df.index)
        df['CST_acc_std'] = pd.Series(0, index=df.index)
        df['CST_runtime'] = pd.Series(0, index=df.index)
    if run_RKT:
        df['MiniRKT_f1_mean'] = pd.Series(0, index=df.index)
        df['MiniRKT_f1_std'] = pd.Series(0, index=df.index)
        df['MiniRKT_acc_mean'] = pd.Series(0, index=df.index)
        df['MiniRKT_acc_std'] = pd.Series(0, index=df.index)
        df['MiniRKT_runtime'] = pd.Series(0, index=df.index)
    if run_MrSEQL:
        df['MrSEQL_f1_mean'] = pd.Series(0, index=df.index)
        df['MrSEQL_f1_std'] = pd.Series(0, index=df.index)
        df['MrSEQL_acc_mean'] = pd.Series(0, index=df.index)
        df['MrSEQL_acc_std'] = pd.Series(0, index=df.index)
        df['MrSEQL_runtime'] = pd.Series(0, index=df.index)
    if run_SFC:
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

pipe_cst = make_pipeline(ConvolutionalShapeletTransformer(P=P,
                                                          n_trees=n_trees,
                                                          max_ft=max_ft,
                                                          n_bins=n_bins,
                                                          random_state=random_state),
                          RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                                           normalize=True))

pipe_sfc = make_pipeline(ShapeletForestClassifier(n_estimators=n_trees,
                                                  metric='scaled_euclidean',
                                                  random_state=random_state))

pipe_MrSEQL = make_pipeline(MrSEQLClassifier(symrep=['sax','sfa']))

# Process the dataset by "size" (n bytes). This is not mandatory to do,
# But you can launch a process going from high to low and another low to high
# by changing the loop iteration.
dataset_size = {}
for name in dataset_names:
    
    ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
        ds_path, 0, normalize=True)
    size = X_train.nbytes+X_test.nbytes
    print("{} : {} bytes".format(name,size))
    dataset_size.update({name:size})
    
dataset_size = {k: v for k, v in sorted(dataset_size.items(), key=lambda item: item[1])}

#Do first run for numba compilations:
ds_path = base_UCR_resamples_path+"{}/{}".format(list(dataset_size.keys())[0], list(dataset_size.keys())[0])
X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
    ds_path, 0, normalize=True)
pipe_cst.fit(X_train, y_train)
pipe_cst.predict(X_test)

for name in dataset_size.keys():
    print(name)
    ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
    splitter = UCR_stratified_resample(n_cv, ds_path)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
        ds_path, 0, normalize=True)
    if run_RKT and df.loc[name, 'MiniRKT_f1_mean'] == 0:
        n_jobs, n_threads = get_n_jobs_n_threads(dataset_size[name], size_mult=2500)
        print("Processing RKT with {} jobs and {} thread".format(n_jobs, n_threads))
        set_num_threads(n_threads)
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_rkt, X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'MiniRKT_acc_mean'] = acc_mean
        df.loc[name, 'MiniRKT_acc_std'] = acc_std
        df.loc[name, 'MiniRKT_f1_mean'] = f1_mean
        df.loc[name, 'MiniRKT_f1_std'] = f1_std
        df.loc[name, 'MiniRKT_runtime'] = time
        df.to_csv(csv_name)

    if run_CST and df.loc[name, 'CST_f1_mean'] == 0:
        n_jobs, n_threads = get_n_jobs_n_threads(dataset_size[name], size_mult=3000)
        print("Processing CST with {} jobs and {} thread".format(n_jobs, n_threads))
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_cst.set_params(convolutionalshapelettransformer_onlyleaves__n_jobs=n_threads),
            X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'CST_acc_mean'] = acc_mean
        df.loc[name, 'CST_acc_std'] = acc_std
        df.loc[name, 'CST_f1_mean'] = f1_mean
        df.loc[name, 'CST_f1_std'] = f1_std
        df.loc[name, 'CST_runtime'] = time
        df.to_csv(csv_name)

    if run_MrSEQL and df.loc[name, 'MrSEQL_f1_mean'] == 0:
        n_jobs, n_threads = get_n_jobs_n_threads(dataset_size[name], size_mult=2500)
        set_num_threads(n_threads)
        print("Processing MrSEQL with {} jobs and {} thread".format(n_jobs, n_threads))
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_MrSEQL, X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'MrSEQL_acc_mean'] = acc_mean
        df.loc[name, 'MrSEQL_acc_std'] = acc_std
        df.loc[name, 'MrSEQL_f1_mean'] = f1_mean
        df.loc[name, 'MrSEQL_f1_std'] = f1_std
        df.loc[name, 'MrSEQL_runtime'] = time
        df.to_csv(csv_name)

    if run_SFC and df.loc[name, 'SFC_f1_mean'] == 0:
        n_jobs, n_threads = get_n_jobs_n_threads(dataset_size[name], size_mult=2500)
        set_num_threads(n_threads)
        print("Processing SFC with {} jobs and {} thread".format(n_jobs, n_threads))
        acc_mean, acc_std, f1_mean, f1_std, time = run_pipeline(
            pipe_sfc.set_params(shapeletforestclassifier__n_jobs=n_threads),
            X_train, X_test, y_train, y_test, splitter, n_jobs)
        df.loc[name, 'SFC_acc_mean'] = acc_mean
        df.loc[name, 'SFC_acc_std'] = acc_std
        df.loc[name, 'SFC_f1_mean'] = f1_mean
        df.loc[name, 'SFC_f1_std'] = f1_std
        df.loc[name, 'SFC_runtime'] = time
        df.to_csv(csv_name)

    print('---------------------')

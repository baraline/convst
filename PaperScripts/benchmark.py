# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from datetime import datetime

from sktime.transformations.panel.rocket import MiniRocket as MiniRKT
from sktime.classification.shapelet_based import MrSEQLClassifier

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV

from CST.utils.dataset_utils import load_sktime_arff_file
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer

from wildboar.ensemble import ShapeletForestClassifier

"""
I did this "dummy" script to have more control over what was included in the 
time measurments. If you want to do things faster, using cross validate pipeline
with parallel jobs could (didn't check source code of sklearn) lead to the same results.

The datasets we use here are not directly available via the sktime interface.
You must download them from the timeserieclassifcation website :
    
Link to full archive http://www.timeseriesclassification.com/Downloads/DucksAndGeese.zip

By placing the _TRAIN.arff and _TEST.arff in the folder specified by the path variable,
 you can simply change the name of the dataset in the functions bellow if you wish to 
 try it on other datasets, be sure in this case to change the lengths that are considered.
 
"""
run_cst = True
run_rkt = True
run_sfc = True
run_sql = True

resume = False

n_jobs = 20
csv_name = 'tslength_Benchmark.csv'
lengths = np.asarray([1e+1, 1e+2, 1e+3, 1e+4, 2.5e+4]).astype(int)
if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
else:
    df = pd.DataFrame(index=lengths)
    df['CST'] = pd.Series(0, index=df.index)
    df['MiniRKT'] = pd.Series(0, index=df.index)
    df['MrSEQL'] = pd.Series(0, index=df.index)
    df['SFC'] = pd.Series(0, index=df.index)


path = r"/home/prof/guillaume/Shapelets/ts_datasets/"
X_train, X_test, y_train, y_test, le = load_sktime_arff_file(
    path+"DucksAndGeese")
n_cv = 10

pipe_rkt = make_pipeline(MiniRKT(),
                         RidgeClassifierCV(alphas=np.logspace(-6, 6, 20),
                                           normalize=True))


pipe_cst = make_pipeline(ConvolutionalShapeletTransformer(P=80, n_trees=100,
                                                          max_ft=1.0, n_bins=11,
                                                          n_jobs=n_jobs),
                         RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), 
                                           normalize=True))

pipe_sfc = make_pipeline(ShapeletForestClassifier(n_jobs=n_jobs))

pipe_MrSEQL = make_pipeline(MrSEQLClassifier(symrep=['sax', 'sfa']))

#Run all script a first time to avoid issues related to first numba run being slower
pipe_cst.fit(X_train[0:100, :, 0:100], y_train[0:100])
pipe_cst.predict(X_train[0:100, :, 0:100])
pipe_sfc.fit(X_train[0:100, :, 0:100], y_train[0:100])
pipe_sfc.predict(X_train[0:100, :, 0:100])
pipe_rkt.fit(X_train[0:100, :, 0:100], y_train[0:100])
pipe_rkt.predict(X_train[0:100, :, 0:100])
pipe_MrSEQL.fit(X_train[0:100, :, 0:100], y_train[0:100])
pipe_MrSEQL.predict(X_train[0:100, :, 0:100])


def time_pipe(pipeline, X_train, y_train, X_test):
    t0 = datetime.now()
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_test)
    t1 = datetime.now()
    return (t1-t0).total_seconds()


for l in lengths:
    x1 = X_train[:, :, :l]
    x2 = X_test[:, :, :l]
    print(x1.shape)
    # CST
    if run_cst and df.loc[l, 'CST'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('CST', l, i_cv))
            timing.append(time_pipe(pipe_cst, x1, y_train, x2))
        df.loc[l, 'CST'] = np.mean(timing)
        df.to_csv(csv_name)

    # RKT
    if run_rkt and df.loc[l, 'MiniRKT'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('MiniRKT', l, i_cv))
            timing.append(time_pipe(pipe_rkt, x1, y_train, x2))
        df.loc[l, 'MiniRKT'] = np.mean(timing)
        df.to_csv(csv_name)

    # MrSEQL
    if run_sql and df.loc[l, 'MrSEQL'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('MrSEQL', l, i_cv))
            timing.append(time_pipe(pipe_MrSEQL, x1, y_train, x2))
        df.loc[l, 'MrSEQL'] = np.mean(timing)
        df.to_csv(csv_name)

    # SFC
    if run_sfc and df.loc[l, 'SFC'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('SFC', l, i_cv))
            timing.append(time_pipe(pipe_sfc, x1, y_train, x2))
        df.loc[l, 'SFC'] = np.mean(timing)
        df.to_csv(csv_name)

# In[]:
resume = False

X_train, X_test, y_train, y_test, le = load_sktime_arff_file(
    path+"InsectSound")
n_classes = np.bincount(y_train).shape[0]
n_per_class = np.asarray([10, 50, 100, 250, 400]).astype(int)
csv_name = 'n_samples_Benchmark.csv'

if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
else:
    df = pd.DataFrame(index=n_per_class*n_classes)
    df['CST'] = pd.Series(0, index=df.index)
    df['MiniRKT'] = pd.Series(0, index=df.index)
    df['MrSEQL'] = pd.Series(0, index=df.index)
    df['SFC'] = pd.Series(0, index=df.index)

n_cv = 10
for n in n_per_class:
    x1 = np.asarray([np.random.choice(np.where(y_train == i)[
                    0], n, replace=False) for i in np.unique(y_train)]).reshape(-1)
    x2 = np.asarray([np.random.choice(np.where(y_test == i)[0],
                                      n, replace=False) for i in np.unique(y_train)]).reshape(-1)
    print(x1.shape)
    if run_cst and df.loc[n*n_classes, 'CST'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('CST', n, i_cv))
            timing.append(
                time_pipe(pipe_cst, X_train[x1], y_train[x1], X_test[x2]))
        df.loc[n*n_classes, 'CST'] = np.mean(timing)
        df.to_csv(csv_name)

    # RKT
    if run_rkt and df.loc[n*n_classes, 'MiniRKT'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('MiniRKT', n, i_cv))
            timing.append(
                time_pipe(pipe_rkt, X_train[x1], y_train[x1], X_test[x2]))
        df.loc[n*n_classes, 'MiniRKT'] = np.mean(timing)
        df.to_csv(csv_name)

    # MrSEQL
    if run_sql and df.loc[n*n_classes, 'MrSEQL'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('MrSEQL', n, i_cv))
            timing.append(
                time_pipe(pipe_MrSEQL, X_train[x1], y_train[x1], X_test[x2]))
        df.loc[n*n_classes, 'MrSEQL'] = np.mean(timing)
        df.to_csv(csv_name)

    # SFC
    if run_sfc and df.loc[n*n_classes, 'SFC'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('SFC', n, i_cv))
            timing.append(
                time_pipe(pipe_sfc, X_train[x1], y_train[x1], X_test[x2]))
        df.loc[n*n_classes, 'SFC'] = np.mean(timing)
        df.to_csv(csv_name)

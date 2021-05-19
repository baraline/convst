# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:51:59 2021

@author: Antoine
"""
from sktime.transformations.panel.rocket import MiniRocket as MiniRKT
from CST.utils.dataset_utils import load_sktime_arff_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import numpy as np
#from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
from CST.shapelet_transforms.try_CST import ConvolutionalShapeletTransformer_tree
from sktime.classification.shapelet_based import MrSEQLClassifier
from sklearn.pipeline import make_pipeline
from datetime import datetime
from wildboar.ensemble import ShapeletForestClassifier

"""
I did this "dummy" script to have more control over what was included in the 
time measurments.

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
csv_name = 'tslength_Benchmark.csv'
lengths = np.asarray([1e+1, 1e+2, 1e+3, 1e+4, 1e+5]).astype(int)
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

#SFC use 100 tree on default
pipe_cst = make_pipeline(ConvolutionalShapeletTransformer_tree(
                                                          P=80,
                                                          n_trees=100,
                                                          max_ft=1.0,
                                                          n_bins=9,
                                                          ),
                         RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))

pipe_sfc = make_pipeline(ShapeletForestClassifier())

pipe_MrSEQL = make_pipeline(MrSEQLClassifier(symrep=['sax','sfa']))


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
n_per_class = np.asarray([10, 50, 100, 250, 500]).astype(int)
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
    print(X_train[x1, 0, :].shape)
    if run_cst and df.loc[n*n_classes, 'CST'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('CST', l, i_cv))
            timing.append(time_pipe(pipe_cst, x1, y_train, x2))
        df.loc[n*n_classes, 'CST'] = np.mean(timing)
        df.to_csv(csv_name)
        
    # RKT
    if run_rkt and df.loc[n*n_classes, 'MiniRKT'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('MiniRKT', l, i_cv))
            timing.append(time_pipe(pipe_rkt, x1, y_train, x2))
        df.loc[n*n_classes, 'MiniRKT'] = np.mean(timing)
        df.to_csv(csv_name)

    # MrSEQL
    if run_sql and df.loc[n*n_classes, 'MrSEQL'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('MrSEQL', l, i_cv))
            timing.append(time_pipe(pipe_MrSEQL, x1, y_train, x2))
        df.loc[n*n_classes, 'MrSEQL'] = np.mean(timing)
        df.to_csv(csv_name)

    # SFC
    if run_sfc and df.loc[n*n_classes, 'SFC'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('SFC', l, i_cv))
            timing.append(time_pipe(pipe_sfc, x1, y_train, x2))
        df.loc[n*n_classes, 'SFC'] = np.mean(timing)
        df.to_csv(csv_name)

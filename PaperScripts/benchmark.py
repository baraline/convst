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
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
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


resume = True
csv_name = 'tslength_Benchmark.csv'
lengths = np.asarray([1e+1,1e+2,1e+3,1e+4,1e+5]).astype(int)
if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
else:
    df = pd.DataFrame(index=lengths)
    df['cst'] = pd.Series(0, index=df.index)
    df['rkt'] = pd.Series(0, index=df.index)
    df['sfc'] = pd.Series(0, index=df.index)

path = r"/home/prof/guillaume/Shapelets/ts_datasets/"
X_train, X_test, y_train, y_test, le = load_sktime_arff_file(path+"DucksAndGeese")
n_cv = 10

for l in lengths:
    x1 = X_train[:,:,:l]
    x2 = X_test[:,:,:l]
    
    #CST
    if df.loc[l, 'cst'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('cst',l,i_cv))
            p = make_pipeline(MiniConvolutionalShapeletTransformer(),
                              RandomForestClassifier())
            t0 = datetime.now()
            p.fit(x1, y_train)
            p.predict(x2)
            t1 = datetime.now()
            timing.append((t1-t0).total_seconds())
        df.loc[l,'cst'] = np.mean(timing)
        df.to_csv(csv_name)
    
    #RKT
    if df.loc[l, 'rkt'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('rkt',l,i_cv))
            p = make_pipeline(MiniRKT(),
                        RidgeClassifierCV(alphas=np.logspace(-4, 4, 10), normalize=True))
            t0 = datetime.now()
            p.fit(x1, y_train)
            p.predict(x2)
            t1 = datetime.now()
            timing.append((t1-t0).total_seconds())
        df.loc[l,'rkt'] = np.mean(timing)
        df.to_csv(csv_name)
    
    #SFC
    if df.loc[l, 'sfc'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('sfc',l,i_cv))
            p = ShapeletForestClassifier()
            t0 = datetime.now()
            p.fit(x1[:,0,:], y_train)
            p.predict(x2[:,0,:])
            t1 = datetime.now()
            timing.append((t1-t0).total_seconds())
        df.loc[l,'sfc'] = np.mean(timing)
        df.to_csv(csv_name)

# In[]:
resume=False
X_train, X_test, y_train, y_test, le = load_sktime_arff_file(path+"InsectSound")
n_classes = np.bincount(y_train).shape[0]
n_per_class = np.asarray([150,300,625,1250,2500]).astype(int)
csv_name = 'n_samples_Benchmark.csv'

if resume:
    df = pd.read_csv(csv_name)
    df = df.set_index('Unnamed: 0')
else:
    df = pd.DataFrame(index=n_per_class*n_classes)
    df['cst'] = pd.Series(0, index=df.index)
    df['rkt'] = pd.Series(0, index=df.index)
    df['sfc'] = pd.Series(0, index=df.index)

n_cv = 10
for n in n_per_class:
    
    x1 = X_train[np.asarray([np.random.choice(np.where(y_train==i)[0],n,replace=False) for i in np.unique(y_train)]).reshape(-1)]
    x2 = X_test[np.asarray([np.random.choice(np.where(y_test==i)[0],n,replace=False) for i in np.unique(y_train)]).reshape(-1)]

    #CST
    if df.loc[n*n_classes, 'cst'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('cst',n,i_cv))
            p = make_pipeline(MiniConvolutionalShapeletTransformer(),
                              RandomForestClassifier(n_estimators=400))
            t0 = datetime.now()
            p.fit(x1, y_train)
            p.predict(x2)
            t1 = datetime.now()
            timing.append((t1-t0).total_seconds())
        df.loc[n*n_classes,'cst'] = np.mean(timing)
        df.to_csv(csv_name)
    
    #RKT
    if df.loc[n*n_classes,'rkt'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('rkt',n,i_cv))
            p = make_pipeline(MiniRKT(),
                        RidgeClassifierCV(alphas=np.logspace(-4, 4, 10), normalize=True))
            t0 = datetime.now()
            p.fit(x1, y_train)
            p.predict(x2)
            t1 = datetime.now()
            timing.append((t1-t0).total_seconds())
        df.loc[n*n_classes,'rkt'] = np.mean(timing)
        df.to_csv(csv_name)
    
    #SFC
    if df.loc[n*n_classes, 'sfc'] == 0:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format('sfc',n,i_cv))
            p = ShapeletForestClassifier(n_estimators=400)
            t0 = datetime.now()
            p.fit(x1[:,0,:], y_train)
            p.predict(x2[:,0,:])
            t1 = datetime.now()
            timing.append((t1-t0).total_seconds())
        df.loc[n*n_classes,'sfc'] = np.mean(timing)
        df.to_csv(csv_name)
        
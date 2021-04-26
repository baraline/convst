# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:21:00 2021

@author: A694772
"""
from CST.base_transformers.rocket import ROCKET
from sktime.transformations.panel.rocket import MiniRocket as MiniRKT
from CST.utils.dataset_utils import load_sktime_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import numpy as np
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer
from datetime import datetime

resume=False

print("Imports OK")
n_cv = 10
use_class_weights=True
n_splits=4
p=[100,90,80]
csv_name = 'CV_{}_results_{}_{}_{}.csv'.format(n_cv,int(use_class_weights),n_splits,p)

if resume:
    df = pd.read_csv(csv_name)
else:
    #df = pd.read_csv(r"C:\Utilisateurs\A694772\Documents\TESTF1_MEANS.csv")
    df = pd.read_csv(r"/home/guillaume/Shapelets/TESTF1_MEANS.csv")
    print("loaded dataset")
    keys = df.columns.difference(['TESTF1'])
    df.drop(keys,axis=1,inplace=True)

    df['MiniCST_mean'] = pd.Series(0, index=df.index)
    df['MiniCST_std'] = pd.Series(0, index=df.index)
    df['RKT_mean'] = pd.Series(0, index=df.index)
    df['RKT_std'] = pd.Series(0, index=df.index)
    df['MiniRKT_mean'] = pd.Series(0, index=df.index)
    df['MiniRKT_std'] = pd.Series(0, index=df.index)
    df['Runtime_MiniCST'] = pd.Series('0', index=df.index)
    df['Runtime_RKT'] = pd.Series('0', index=df.index)
    df['Runtime_MiniRKT'] = pd.Series('0', index=df.index)
    df['MiniCST_n_shp'] = pd.Series(0, index=df.index)
    df['MiniCST_n_shp_used'] = pd.Series(0, index=df.index)
    df['MiniCST_n_kernel'] = pd.Series(0, index=df.index)
    
for name in df['TESTF1'].values:
    print(name)
    mask = df[df['TESTF1']==name].index
    X, y, le = load_sktime_dataset(name,normalize=True)
    if df.loc[mask,'MiniCST_mean'].values[0] == 0 and X.shape[0]<=2000 and X.shape[2]<=3000 and all(np.bincount(y)>=n_cv):
        pipe_rkt = make_pipeline(ROCKET(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))
        cv = cross_validate(pipe_rkt, X, y, cv=n_cv, scoring={'f1':make_scorer(f1_score, average='macro')},n_jobs=None)
        print("F1-Score for ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
        df.loc[mask,'RKT_mean'] = np.mean(cv['test_f1'])
        df.loc[mask,'RKT_std'] = np.std(cv['test_f1'])
        df.loc[mask,'Runtime_RKT'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        pipe_rkt = make_pipeline(MiniRKT(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))
        cv = cross_validate(pipe_rkt, X, y, cv=n_cv, scoring={'f1':make_scorer(f1_score, average='macro')},n_jobs=None)
        print("F1-Score for MINI-ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
        df.loc[mask,'MiniRKT_mean'] = np.mean(cv['test_f1'])
        df.loc[mask,'MiniRKT_std'] = np.std(cv['test_f1'])
        df.loc[mask,'Runtime_MiniRKT'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        #We must use StratifiedShuffleSplit instead to cross_validate to extract some parameters to build some plots
        sss = StratifiedShuffleSplit(n_splits=n_cv, test_size=1/n_cv, random_state=0)
        f1 = []
        n_shp = []
        n_shp_used = []
        n_k = []
        time = []
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            rf = RandomForestClassifier(n_estimators=400)
            mCST = MiniConvolutionalShapeletTransformer()
            d0 = datetime.now()
            mCST.fit(X_train, y_train, use_class_weights=use_class_weights,
                     n_splits=n_splits, p=p)
            X_train_shp = mCST.transform(X_train)
            X_test_shp = mCST.transform(X_test)
            rf.fit(X_train_shp, y_train)
            pred = rf.predict(X_test_shp)
            d1 = datetime.now()
            n_shp.append(mCST.n_shapelets)
            n_k.append(mCST.n_kernels)
            n_shp_used.append(np.where(rf.feature_importances_>0.000001)[0].shape[0])
            f1.append(f1_score(y_test,pred,average='macro'))
            time.append((d1-d0).total_seconds())
        
        print("F1-Score for MiniCST RF : {}".format(np.mean(f1)))
        
        df.loc[mask,'MiniCST_mean'] = np.mean(f1)
        df.loc[mask,'MiniCST_std'] = np.std(f1)
        df.loc[mask,'Runtime_MiniCST'] = np.mean(time)
        df.loc[mask,'MiniCST_n_shp'] = np.mean(n_shp)
        df.loc[mask,'MiniCST_n_shp_used'] = np.mean(n_shp_used)
        df.loc[mask,'MiniCST_n_kernel'] = np.mean(n_k)
            
        df.to_csv(csv_name)
        print('---------------------')
    else:
        df = df.drop(mask,axis=0)

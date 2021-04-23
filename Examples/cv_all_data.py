# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:21:00 2021

@author: A694772
"""
from CST.base_transformers.rocket import ROCKET
from CST.base_transformers.minirocket import MiniRocket
from CST.utils.dataset_utils import load_sktime_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import numpy as np
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer

resume=False

print("Imports OK")


if resume:
    df = pd.read_csv("CV_results_all.csv")
else:
    #df = pd.read_csv(r"C:\Utilisateurs\A694772\Documents\TESTF1_MEANS.csv")
    df = pd.read_csv(r"/home/guillaume/Shapelets/TESTF1_MEANS.csv")
    print("loaded dataset")
    keys = ["ProximityForest","WEASEL","S-BOSS","BOSS","cBOSS","TSF",
            "RISE","ResNet","InceptionTime","TS-CHIEF","HIVE-COTE v1.0"]
    df.drop(keys,axis=1,inplace=True)
    
    df['MiniCST'] = pd.Series(0, index=df.index)
    df['RKT'] = pd.Series(0, index=df.index)
    df['MiniRKT'] = pd.Series(0, index=df.index)
    df['Runtime_MiniCST'] = pd.Series('0', index=df.index)
    df['Runtime_RKT'] = pd.Series('0', index=df.index)
    df['Runtime_MiniRKT'] = pd.Series('0', index=df.index)

for name in df['TESTF1'].values:
    print(name)
    mask = df[df['TESTF1']==name].index
    X, y, le = load_sktime_dataset(name,normalize=True)
    if df.loc[mask,'MiniCST'].values[0] == 0 and X.shape[0]<=600 and X.shape[2]<=1300:
        pipe_rkt = make_pipeline(ROCKET(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))
        cv = cross_validate(pipe_rkt, X, y, cv=10, scoring={'f1':make_scorer(f1_score, average='macro')},n_jobs=None)
        print("F1-Score for ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
        df.loc[mask,'RKT'] = np.mean(cv['test_f1'])
        df.loc[mask,'Runtime_RKT'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        pipe_rkt = make_pipeline(MiniRocket(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))
        cv = cross_validate(pipe_rkt, X, y, cv=10, scoring={'f1':make_scorer(f1_score, average='macro')},n_jobs=None)
        print("F1-Score for MINI-ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
        df.loc[mask,'MiniRKT'] = np.mean(cv['test_f1'])
        df.loc[mask,'Runtime_MiniRKT'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        pipe_cst = make_pipeline(MiniConvolutionalShapeletTransformer(),
                             RandomForestClassifier(n_estimators=400))
        
        cv = cross_validate(pipe_cst, X, y, cv=10, scoring={'f1':make_scorer(f1_score, average='macro')},n_jobs=None)
        print("F1-Score for MiniCST RF : {}".format(np.mean(cv['test_f1'])))
        df.loc[mask,'MiniCST'] = np.mean(cv['test_f1'])
        df.loc[mask,'Runtime_MiniCST'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        df.to_csv('CV_10_results_mini_no_norm_no_comb.csv')
        print('---------------------')
    else:
        df = df.drop(mask,axis=0)

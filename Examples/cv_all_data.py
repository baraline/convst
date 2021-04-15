# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:21:00 2021

@author: A694772
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:18:40 2021

@author: Antoine
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:21:39 2021

@author: A694772
"""
print("Does this even work ?")
from CST.base_transformers.rocket import ROCKET
from CST.utils.dataset_utils import load_sktime_dataset
from sklearn.metrics import f1_score
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
import pandas as pd
import numpy as np
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
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
            "RISE","ResNet"]
    df.drop(keys,axis=1,inplace=True)
    
    df['CST'] = pd.Series(0, index=df.index)
    df['n_shp_CST'] = pd.Series(0, index=df.index)
    df['RKT'] = pd.Series(0, index=df.index)
    df['Runtime_CST'] = pd.Series('0', index=df.index)
    df['Runtime_RKT'] = pd.Series('0', index=df.index)

for name in df['TESTF1'].values:
    print(name)
    mask = df[df['TESTF1']==name].index
    if df.loc[mask,'CST'].values[0] == 0:
        X, y, le = load_sktime_dataset(name,normalize=True)
        
        pipe_rkt = make_pipeline(ROCKET(),
                             RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True))
        cv = cross_validate(pipe_rkt, X, y, cv=10, scoring={'f1':make_scorer(f1_score, average='macro')},n_jobs=-1)
        print("F1-Score for ROCKET Ridge : {}".format(np.mean(cv['test_f1'])))
        df.loc[mask,'RKT'] = np.mean(cv['test_f1'])
        df.loc[mask,'Runtime_RKT'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        
        pipe_cst = make_pipeline(ConvolutionalShapeletTransformer(),
                             RandomForestClassifier(n_estimators=400))
        
        cv = cross_validate(pipe_cst, X, y, cv=10, scoring={'f1':make_scorer(f1_score, average='macro')},n_jobs=-1)
        print("F1-Score for CST RF : {}".format(np.mean(cv['test_f1'])))
        df.loc[mask,'CST'] = np.mean(cv['test_f1'])
        df.loc[mask,'Runtime_CST'] = np.mean(cv['fit_time'] + cv['score_time'])
        
        df.to_csv('CV_10_results.csv')
        print('---------------------')

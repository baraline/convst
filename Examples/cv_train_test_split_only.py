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
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import f1_score
from datetime import datetime
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer

resume=True

print("Imports OK")
#TODO update this script
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

t_start_all = datetime.now()
for name in df['TESTF1'].values:
    print(name)
    mask = df[df['TESTF1']==name].index
    if df.loc[mask,'CST'].values[0] == 0:
        X_train, X_test, y_train, y_test, le = load_sktime_dataset_split(name,normalize=True)
        t_start = datetime.now()
        
        rkt = ROCKET()
        X_rkt_train = rkt.fit_transform(X_train)
        X_rkt_test = rkt.transform(X_test)
        rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True, class_weight="balanced").fit(X_rkt_train, y_train)
        pred = rdg.predict(X_rkt_test)
        t_end = datetime.now()
        
        
        score = f1_score(y_test, pred, average='macro')
        print("F1-Score for ROCKET RF : {}".format(score))
        df.loc[mask,'RKT'] = score
        
        df.loc[mask,'Runtime_RKT'] = str((t_end - t_start)/60)
    
        t_start = datetime.now()
        
        cst = ConvolutionalShapeletTransformer(verbose=1)
        cst.fit(X_train, y_train, n_shapelet_per_combination=2, n_iter_per_comb=5, n_bins=7, percentile_select=90)
        X_cst_train = cst.transform(X_train)
        X_cst_test = cst.transform(X_test)    
        rf = RandomForestClassifier(n_estimators=400).fit(X_cst_train, y_train)
        pred = rf.predict(X_cst_test)    
        
        t_end = datetime.now()
        
        score = f1_score(y_test, pred, average='macro')
        df.loc[mask,'CST'] = score
        df.loc[mask,'n_shp_CST'] = rf.feature_importances_[rf.feature_importances_ > 0].size
        
        df.loc[mask,'Runtime_CST'] = str((t_end - t_start)/60)
        
        print("F1-Score for CST RF : {}".format(score))
        df.to_csv('CV_split_results.csv')
        print('---------------------')

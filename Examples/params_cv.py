# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:30:45 2021

@author: A694772
"""
from CST.utils.dataset_utils import load_sktime_dataset
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from sklearn.pipeline import Pipeline
from itertools import combinations
from sklearn.model_selection import GridSearchCV

resume=Falseconvoconv

ps = [[95]]
for r in range(1,5):
    ps.extend(list(combinations([100,95,90,85,80],r)))
n_splits = [1,3,5,7]
params = {'CST__p':ps,
          'CST__n_splits':n_splits}
print(params)

if resume:
    df = pd.read_csv('params_csv.csv')
else:
    df = pd.DataFrame()

datasets = "BirdChicken,BeetleFly,Beef,Car,CricketX,CricketY,CricketZ,DistalPhalanxTW,FiftyWords,Fish,Haptics,Herring,ItalyPowerDemand,Meat,MedicalImages,MiddlePhalanxOutlineAgeGroup,MiddlePhalanxOutlineCorrect,SwedishLeaf,OliveOil,OSULeaf,Yoga,Worms,UWaveGestureLibraryY,Trace,ShapeletSim"
for dataset_name in datasets.split(','):
    results = {}
    X, y, le = load_sktime_dataset(dataset_name,normalize=True)
    print(dataset_name)
    if dataset_name not in df.index.values():
        pipe = Pipeline([('CST',MiniConvolutionalShapeletTransformer()),
                             ('rf',RandomForestClassifier(n_estimators=400))])
        clf = GridSearchCV(pipe, params)
        clf.fit(X, y)
        p_key = clf.cv_results_['params']
        rank = clf.cv_results_['rank_test_score']
        for i, p in enumerate(p_key):
            if str(p) in results.keys():
                results[str(p)].append(rank[i])
            else:
                results.update({str(p):[rank[i]]})
        
        df = pd.concat([df, pd.DataFrame(results, index=[dataset_name])],axis=0)
        df.to_csv('params_csv.csv',sep=';')
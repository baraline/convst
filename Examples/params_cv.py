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

resume = False

available_memory_bytes = 40*1e9
max_cpu_cores = 20
ps = []
for r in range(1, 5):
    ps.extend(list(combinations([100, 95, 90, 85, 80], r)))
n_splits = [1, 3, 5, 7, 10]
params = {'CST__P': ps,
          'CST__n_splits': n_splits}
print(params)

if resume:
    df = pd.read_csv('params_csv2.csv', sep=';')
    df = df.set_index('Unnamed: 0')
    print(df.index.values)
else:
    df = pd.DataFrame()

datasets = "BirdChicken,BeetleFly,Beef,Car,CricketX,CricketY,CricketZ,DistalPhalanxTW,FiftyWords,Fish,Haptics,Herring,ItalyPowerDemand,Meat,MedicalImages,MiddlePhalanxOutlineAgeGroup,MiddlePhalanxOutlineCorrect,SwedishLeaf,OliveOil,OSULeaf,Yoga,Worms,UWaveGestureLibraryY,Trace,ShapeletSim"
dataset_names = datasets.split(',')
dsizes = []
for dataset_name in dataset_names:
    X, y, le = load_sktime_dataset(dataset_name, normalize=True)
    dsizes.append(X.nbytes*4680)
dsizes = np.asarray(dsizes)
print(dsizes)

for i_dataset in np.argsort(dsizes):
    results = {}
    print(dataset_names[i_dataset])
    if dataset_names[i_dataset] not in df.index.values:
        X, y, le = load_sktime_dataset(
            dataset_names[i_dataset], normalize=True)
        pipe = Pipeline([('CST', MiniConvolutionalShapeletTransformer()),
                         ('rf', RandomForestClassifier(n_estimators=400))])

        n_jobs = available_memory_bytes // X.nbytes
        n_jobs = max(n_jobs if n_jobs <= max_cpu_cores else max_cpu_cores, 1)
        print('Launching {} parallel jobs'.format(n_jobs))
        clf = GridSearchCV(pipe, params, n_jobs=n_jobs, cv=10, verbose=1)
        clf.fit(X, y)
        print('Done')
        p_key = clf.cv_results_['params']
        rank = clf.cv_results_['rank_test_score']
        for i, p in enumerate(p_key):
            if str(p) in results.keys():
                results[str(p)].append(rank[i])
            else:
                results.update({str(p): [rank[i]]})
        print('Dumping results to csv')
        df = pd.concat(
            [df, pd.DataFrame(results, index=[dataset_names[i_dataset]])], axis=0)
        df.to_csv('params_csv2.csv', sep=';')

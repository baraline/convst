# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:02:04 2021

@author: A694772
"""
import pandas as pd
import numpy as np
import convst

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from convst.utils import load_sktime_arff_file_resample_id, UCR_stratified_resample
from convst.transformers.convolutional_ST import ConvolutionalShapeletTransformer

n_trees=100
max_ft=1.0
P=80
n_bins=11

version = convst.__version__

df_path = r"/home/prof/guillaume/convst/df_perf.csv"
#df = pd.DataFrame(columns=["Algorithm","Version","Dataset","F1_score","Accuracy_score","RunTime"])
df = pd.read_csv(df_path,index_col=0)
base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"

#Historic "Worse" datasets compared to Rocket.
names = ['Adiac','ShapeletSim','MedicalImages','Worms','Lightning2','EOGHorizontalSignal','EOGVerticalSignal','FreezerSmallTrain','WordSynonyms']

def run_pipeline(pipeline, name, n_jobs):
    ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
        ds_path, 0, normalize=True)
    splitter = UCR_stratified_resample(10, ds_path)
    
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    cv = cross_validate(pipeline, X, y, cv=splitter, n_jobs=n_jobs,
                        scoring={'f1': make_scorer(f1_score, average='macro'),
                                 'acc':make_scorer(accuracy_score)})
    return np.mean(cv['test_acc']),  np.std(cv['test_acc']), np.mean(cv['test_f1']),  np.std(cv['test_f1']), np.mean(cv['fit_time'] + cv['score_time']), np.std(cv['fit_time'] + cv['score_time'])


pipe_cst_rdg = make_pipeline(ConvolutionalShapeletTransformer(P=P,
                                                      n_trees=n_trees,
                                                      max_ft=max_ft,
                                                      n_bins=n_bins,
                                                      n_jobs=5),
                         RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), 
                                           normalize=True))

pipe_cst_rf = make_pipeline(ConvolutionalShapeletTransformer(P=P,
                                                      n_trees=n_trees,
                                                      max_ft=max_ft,
                                                      n_bins=n_bins,
                                                      n_jobs=5),
                         RandomForestClassifier(n_estimators=n_trees,n_jobs=5))

for name in names:
    print("Processing {}".format(name))
    mean_acc, std_acc, mean_f1, std_f1, mean_runtimes, std_runtimes = run_pipeline(pipe_cst_rdg, name, n_jobs=5)
    idx = np.where(np.all(df.values[:,0:3] == np.array(["CST + Ridge",version,name]),axis=1))[0]
    if idx.shape[0] > 0: 
        idx = idx[0]
    else:
        idx = df.shape[0]
    df.loc[idx] = ["CST + Ridge",
                    version,
                    name,
                    str(mean_f1)[0:5] + '(+/- ' + str(std_f1)[0:5] + ')',
                    str(mean_acc)[0:5] + '(+/- ' + str(std_acc)[0:5] + ')',
                    str(mean_runtimes)[0:5] + '(+/- ' + str(std_runtimes)[0:5] + ')']
 
    idx = np.where(np.all(df.values[:,0:3] == np.array(["CST + RF",version,name]),axis=1))[0]
    if idx.shape[0] > 0: 
        idx = idx[0]
    else:
        idx = df.shape[0]
    mean_acc, std_acc, mean_f1, std_f1, mean_runtimes, std_runtimes = run_pipeline(pipe_cst_rf, name, n_jobs=5)
    df.loc[idx] = ["CST + RF",
                    version,
                    name,
                    str(mean_f1)[0:5] + '(+/- ' + str(std_f1)[0:5] + ')',
                    str(mean_acc)[0:5] + '(+/- ' + str(std_acc)[0:5] + ')',
                    str(mean_runtimes)[0:5] + '(+/- ' + str(std_runtimes)[0:5] + ')']
    
    df.to_csv(df_path)


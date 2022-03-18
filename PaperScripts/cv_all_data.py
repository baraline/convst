# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils.dataset_utils import (return_all_dataset_names,
    load_sktime_arff_file_resample_id)
from convst.utils.experiments_utils import UCR_stratified_resample, run_pipeline
from convst.transformers import R_DST, R_DST_NN

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Modify this to your path to the UCR resamples, check README for how to get them. 
# Another splitter is also provided in dataset_utils to make random resamples

base_UCR_resamples_path = r"/home/prof/guillaume/Shapelets/resamples/"
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

print("Imports OK")
#n_cv = 1 to test only on original train test split.
n_cv=30

csv_name = 'CV_{}_results_Random_final_(5_10)_2.csv'.format(
    n_cv)

dataset_names = return_all_dataset_names()

#Initialize result dataframe. This script will also launch RDST without any normalization for comparison, hence the *2
df = pd.DataFrame(0, index=np.arange(dataset_names.shape[0]*2), columns=['dataset','model','acc_mean','acc_std','f1_mean','f1_std','time_mean','time_std'])
df.to_csv(csv_name)


i_df=0
for name in dataset_names:
    print(name)
    ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
    splitter = UCR_stratified_resample(n_cv, ds_path)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
        ds_path, 0, normalize=True
    )
    
    pipeline_RDST = make_pipeline(R_DST(n_shapelets=10000), 
                                     StandardScaler(with_mean=False),
                                     RidgeClassifierCV(np.logspace(-6,6,20)))
    pipeline_RDST_Nn = make_pipeline(R_DST_NN(n_shapelets=10000), 
                                     StandardScaler(with_mean=False),
                                     RidgeClassifierCV(np.logspace(-6,6,20)))

    acc_mean, acc_std, f1_mean, f1_std, time_mean, time_std = run_pipeline(
        pipeline_RDST, X_train, X_test, y_train, y_test, splitter, 1)
    df.loc[i_df, 'acc_mean'] = acc_mean
    df.loc[i_df, 'acc_std'] = acc_std
    df.loc[i_df, 'f1_mean'] = f1_mean
    df.loc[i_df, 'f1_std'] = f1_std
    df.loc[i_df, 'time_mean'] = time_mean
    df.loc[i_df, 'time_std'] = time_std
    df.loc[i_df, 'dataset'] = name
    df.loc[i_df, 'model'] = 'RDST'
    i_df+=1
    df.to_csv(csv_name)
    
    acc_mean, acc_std, f1_mean, f1_std, time_mean, time_std = run_pipeline(
        pipeline_RDST_Nn, X_train, X_test, y_train, y_test, splitter, 1)
    df.loc[i_df, 'acc_mean'] = acc_mean
    df.loc[i_df, 'acc_std'] = acc_std
    df.loc[i_df, 'f1_mean'] = f1_mean
    df.loc[i_df, 'f1_std'] = f1_std
    df.loc[i_df, 'time_mean'] = time_mean
    df.loc[i_df, 'time_std'] = time_std
    df.loc[i_df, 'dataset'] = name
    df.loc[i_df, 'model'] = 'RDST NN'
    i_df+=1
    df.to_csv(csv_name)

    print('---------------------')

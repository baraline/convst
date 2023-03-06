# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils.dataset_utils import (
    return_all_univariate_dataset_names, load_sktime_arff_file_resample_id
)
from convst.utils.experiments_utils import ARFF_stratified_resample, run_pipeline
from convst.classifiers import R_DST_Ensemble


print("Imports OK")
#n_cv = 1 to test only on original train test split.
n_cv=30
# -1 to use all available cores.
n_jobs = -1
# Use true if the script crashed and you want to resume experiment to last saved results.
resume = False
#Specify the path where the resamples are located
base_UCR_resamples_path = r"/home/prof/guillaume/sktime_resamples/"

csv_name = 'CV_{}_results_ensemble.csv'.format(
    n_cv)

dataset_names = return_all_univariate_dataset_names()


#You can add other models here to compare them on the same resamples.
dict_models = {
    "R_DST_Ensemble": R_DST_Ensemble,
}

#Initialize result dataframe.
if not resume:
    df = pd.DataFrame(0, index=np.arange(dataset_names.shape[0]*len(dict_models)), 
         columns=['dataset','model','acc_mean','acc_std',
        'f1_mean','f1_std','time_mean','time_std']
    )
    df.to_csv(csv_name)
else:
    df = pd.read_csv(csv_name, index_col=0)

print(df)

#Numba compilations.
for model_name, model_class in dict_models.items():
    print("Compiling {}".format(model_name))
    X = np.random.rand(5,1,50)
    y = np.array([0,0,1,1,1])
    model_class(n_shapelets_per_estimator=1).fit(X,y).predict(X)
i_df=0

for name in dataset_names:
    print(name)
    ds_path = base_UCR_resamples_path+"{}/{}".format(name, name)
    splitter = ARFF_stratified_resample(n_cv, ds_path)
    X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(
        ds_path, 0, normalize=True
    )
    
    for model_name, model_class in dict_models.items():
        #Condition bellow is used when you want to resume the script and load an existing result dataframe
        if pd.isna(df.loc[i_df, 'acc_mean']) or df.loc[i_df, 'acc_mean'] == None or df.loc[i_df, 'acc_mean'] == 0.0:
            print(model_name)
            pipeline = model_class(n_jobs=n_jobs)
            acc_mean, acc_std, f1_mean, f1_std, time_mean, time_std = run_pipeline(
                pipeline, X_train, X_test, y_train, y_test, splitter, n_jobs=1)
            df.loc[i_df, 'acc_mean'] = acc_mean
            df.loc[i_df, 'acc_std'] = acc_std
            df.loc[i_df, 'f1_mean'] = f1_mean
            df.loc[i_df, 'f1_std'] = f1_std
            df.loc[i_df, 'time_mean'] = time_mean
            df.loc[i_df, 'time_std'] = time_std
            df.loc[i_df, 'dataset'] = name
            df.loc[i_df, 'model'] = model_name
            df.to_csv(csv_name)
        else:
            print('Skipping {} : {}'.format(model_name, df.loc[i_df, 'acc_mean']))
            
        i_df+=1
    print('---------------------')
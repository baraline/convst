# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils.dataset_utils import (return_all_multivariate_dataset_names,
    load_sktime_arff_file_resample_id, load_sktime_dataset_split)
from convst.utils.experiments_utils import cross_validate_UCR_UEA
from sklearn.pipeline import make_pipeline
from convst.classifiers import MR_DST_Ensemble

print("Imports OK")
#n_cv = 1 to test only on original train test split.
n_cv=30

csv_name = 'CV_{}_results_multivariate_ensemble.csv'.format(
    n_cv)

dataset_names = return_all_multivariate_dataset_names()

dict_models = {
    "R_DST_Ensemble": MR_DST_Ensemble,
}
resume=False

#Initialize result dataframe
if resume:
    df = pd.read_csv(csv_name, index_col=0)
else:
    df = pd.DataFrame(0, index=np.arange(dataset_names.shape[0]*len(dict_models)), columns=['dataset','model','acc_mean','acc_std','f1_mean','f1_std','time_mean','time_std'])
    df.to_csv(csv_name)

for model_name, model_class in dict_models.items():
    print("Compiling {}".format(model_name))
    X = np.random.rand(5,3,50)
    y = np.array([0,0,1,1,1])
    model_class(n_shapelets_per_estimator=1).fit(X,y).predict(X)

i_df=0
for name in dataset_names:
    print(name)
    for model_name, model_class in dict_models.items():
        if pd.isna(df.loc[i_df, 'acc_mean']) or df.loc[i_df, 'acc_mean'] == 0.0:
            pipeline = model_class(n_jobs=3, n_jobs_rdst=95//3)
            _scores = cross_validate_UCR_UEA(n_cv, name).score(pipeline)
            df.loc[i_df, 'acc_mean'] = _scores['accuracy'].mean()
            df.loc[i_df, 'acc_std'] = _scores['accuracy'].std()
            df.loc[i_df, 'time_mean'] = _scores['time'].mean()
            df.loc[i_df, 'time_std'] = _scores['time'].std()
            df.loc[i_df, 'dataset'] = name
            df.loc[i_df, 'model'] = model_name
            df.to_csv(csv_name)
        else:
            print('Skipping {} : {}'.format(model_name, df.loc[i_df, 'acc_mean']))
        i_df+=1
    print('---------------------')
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils.dataset_utils import return_all_dataset_names, return_all_univariate_dataset_names
from convst.utils.experiments_utils import cross_validate_UCR_UEA

from convst.classifiers import R_DST_Ensemble, R_DST_Ridge

print("Imports OK")
#n_cv = 1 to test only on original train test split.
n_cv = 30
n_jobs = -1
csv_name = 'CV_{}_results_default.csv'.format(
    n_cv)

# List of datasets to test, here, use all datasets ones, (univariate,
# multivariate, variable length, etc...) see dataset_utils for other choices.
# e.g. to test on all datasets, change to :
# dataset_names = return_all_dataset_names()
dataset_names = return_all_univariate_dataset_names()

# List of models to test
dict_models = {
    "R_DST": R_DST_Ridge,
    "R_DST_Ensemble": R_DST_Ensemble
}

resume=False
#Initialize result dataframe
if resume:
    df = pd.read_csv(csv_name, index_col=0)
else:
    df = pd.DataFrame(0, index=np.arange(dataset_names.shape[0]*len(dict_models)),
                      columns=['dataset','model','acc_mean','acc_std',
                               'time_mean','time_std']
    )
    df.to_csv(csv_name)

for model_name, model_class in dict_models.items():
    print("Compiling {}".format(model_name))
    X = np.random.rand(5,3,50)
    y = np.array([0,0,1,1,1])
    if model_name == 'R_DST_Ensemble':
        model_class(n_shapelets_per_estimator=1).fit(X,y).predict(X)
    if model_name == 'R_DST_Ridge':
        model_class(n_shapelets=1).fit(X,y).predict(X)

i_df=0
for name in dataset_names:
    print(name)
    for model_name, model_class in dict_models.items():
        print(model_name)
        if pd.isna(df.loc[i_df, 'acc_mean']) or df.loc[i_df, 'acc_mean'] == 0.0:
            pipeline = model_class(
                n_jobs=n_jobs
            )
            
            #By default, we use accuracy as score, but other scorer can be passed
            #as parameters (e.g. by default scorers={"accuracy":accuracy_score})
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
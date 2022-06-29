# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from convst.utils.dataset_utils import (return_all_dataset_names,
    load_sktime_arff_file_resample_id, load_sktime_dataset_split)
from convst.utils.experiments_utils import stratified_resample, run_pipeline
from convst.classifiers import RotationForest
from convst.transformers import R_DST
from convst.transformers.dummies import (R_DST_NL, R_ST_NL ,R_ST, R_DST_FT, 
                                         R_DST_CID, R_DST_EUC, R_DST_PH_CID, R_DST_PH)
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler


print("Imports OK")
#n_cv = 1 to test only on original train test split.
n_cv=3

csv_name = 'CV_{}_results_dummies.csv'.format(
    n_cv)

dataset_names = return_all_dataset_names()

#Initialize result dataframe. This script will also launch RDST without any normalization for comparison, hence the *2
df = pd.DataFrame(0, index=np.arange(dataset_names.shape[0]*2), columns=['dataset','model','acc_mean','acc_std','f1_mean','f1_std','time_mean','time_std'])
df.to_csv(csv_name)

dict_models = {
    "R_DST": R_DST,
    "R_DST_NL":R_DST_NL,
    "R_ST_NL":R_ST_NL,
    "R_ST":R_ST,
    "R_DST_FT":R_DST_FT,
    "R_DST_PH":R_DST_PH,
    "R_DST_CID":R_DST_CID,
    "R_DST_EUC":R_DST_EUC,
    "R_DST_PH_CID":R_DST_PH_CID
}
for model_name, model_class in dict_models.items():
    print("Compiling {}".format(model_name))
    X = np.random.rand(5,1,50)
    y = np.array([0,0,1,1,1])
    model_class(n_shapelets=2).fit_transform(X,y)

i_df=0
for name in dataset_names:
    print(name)
    X_train, X_test, y_train, y_test, _= load_sktime_dataset_split(name, normalize=True)
    splitter = stratified_resample(n_cv, X_train.shape[0])
    
    for model_name, model_class in dict_models.items():
        print(model_name)
        pipeline_RDST = make_pipeline(
            model_class(n_jobs=85), 
            StandardScaler(with_mean=False),
            RotationForest(n_estimators=85, min_group=10, max_group=20, n_jobs=85)
        )
    
        acc_mean, acc_std, f1_mean, f1_std, time_mean, time_std = run_pipeline(
            pipeline_RDST, X_train, X_test, y_train, y_test, splitter, n_jobs=1)
        df.loc[i_df, 'acc_mean'] = acc_mean
        df.loc[i_df, 'acc_std'] = acc_std
        df.loc[i_df, 'f1_mean'] = f1_mean
        df.loc[i_df, 'f1_std'] = f1_std
        df.loc[i_df, 'time_mean'] = time_mean
        df.loc[i_df, 'time_std'] = time_std
        df.loc[i_df, 'dataset'] = name
        df.loc[i_df, 'model'] = model_name
        i_df+=1
        df.to_csv(csv_name)

    print('---------------------')

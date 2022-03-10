# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

from convst.utils.dataset_utils import load_sktime_dataset_split
from convst.transformers import R_DST

from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.interval_based import DrCIF
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.transformations.panel.rocket import Rocket

from timeit import default_timer as timer

# Define timining function
def time_pipe(pipeline, X_train, y_train):
    t0 = timer()
    pipeline.fit(X_train, y_train)
    t1 = timer()
    return t1-t0

# Number of validation step
n_cv = 10
# Number of parallel threads for each method
n_jobs=90

pipe_RDST = make_pipeline(
    R_DST(n_jobs=n_jobs),
    StandardScaler(with_mean=False),
    RidgeClassifierCV()
)

pipe_RKT = make_pipeline(
    Rocket(n_jobs=n_jobs),
    StandardScaler(with_mean=False),
    RidgeClassifierCV()
)

models = {'RDST':pipe_RDST, 'Rocket':pipe_RKT,
          'DrCIF':DrCIF(n_jobs=n_jobs),'HC1':HIVECOTEV1(n_jobs=n_jobs),
          'STC':ShapeletTransformClassifier(n_jobs=n_jobs),
          'HC2':HIVECOTEV2(n_jobs=n_jobs)}

# Execute all model once for possible numba compilations
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split("SmoothSubspace")
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
for name in models:
    time_pipe(clone(models[name]), X_train, y_train)

# In[Samples benchmarks]:
csv_name = 'n_samples_benchmarks.csv'    

X_train, X_test, y_train, y_test, le = load_sktime_dataset_split("Crop")
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

stp = X_train.shape[0]//8
lengths = np.arange(stp,(X_train.shape[0])+stp,stp)
df = pd.DataFrame(index=lengths)
df['RDST'] = pd.Series(0, index=df.index)
df['Rocket'] = pd.Series(0, index=df.index)
df['DrCIF'] =  pd.Series(0, index=df.index)
df['HC1'] = pd.Series(0, index=df.index)
df['HC2'] = pd.Series(0, index=df.index)
df['STC'] = pd.Series(0, index=df.index)


from sklearn.utils import resample

for l in lengths:
    x1 = resample(X_train, replace=False, n_samples=l, stratify=y_train, random_state=0)
    y1 = resample(y_train, replace=False, n_samples=l, stratify=y_train, random_state=0)
    print(x1.shape)
    for name in models:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format(name, l, i_cv))
            mod = clone(models[name])
            timing.append(time_pipe(mod, x1, y1))
        df.loc[l, name] = np.mean(timing)
        df.loc[l, name+'_std'] = np.std(timing)
        df.to_csv(csv_name)

# In[Timepoints benchmarks]:
csv_name = 'n_timepoints_benchmarks_HC2.csv'    

X_train, X_test, y_train, y_test, le = load_sktime_dataset_split("Rock")
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

stp = X_train.shape[2]//8
lengths = np.arange(stp,X_train.shape[2]+stp,stp)
df = pd.DataFrame(index=lengths)
df['RDST'] = pd.Series(0, index=df.index)
df['Rocket'] = pd.Series(0, index=df.index)
df['DrCIF'] =  pd.Series(0, index=df.index)
df['HC1'] = pd.Series(0, index=df.index)
df['HC2'] = pd.Series(0, index=df.index)
df['STC'] = pd.Series(0, index=df.index)

for l in lengths:
    x1 = X_train[:,:,:l]
    print(x1.shape)
    for name in models:
        timing = []
        for i_cv in range(n_cv):
            print("{}/{}/n_cv:{}".format(name, l, i_cv))
            mod = clone(models[name])
            timing.append(time_pipe(mod, x1, y_train))
        df.loc[l, name] = np.mean(timing)
        df.loc[l, name+'_std'] = np.std(timing)
        df.to_csv(csv_name)

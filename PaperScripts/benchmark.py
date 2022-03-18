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
X_train, _, y_train, _, _ = load_sktime_dataset_split("SmoothSubspace")
for name in models:
    time_pipe(clone(models[name]), X_train, y_train)

# In[Samples benchmarks]:
csv_name = 'n_samples_benchmarks.csv'    

X_train, _, y_train, _, _ = load_sktime_dataset_split("Crop")

#Had to cut number of samples to get results on our cluster.
n_samples = X_train.shape[0]//4

stp = n_samples//8
lengths = np.arange(stp,(n_samples)+stp,stp)
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
csv_name = 'n_timepoints_benchmarks.csv'    

X_train, _, y_train, _, _ = load_sktime_dataset_split("Rock")
#Had to cut number of samples to get results on our cluster.
n_timestamps = X_train.shape[2]

stp = n_timestamps//8
lengths = np.arange(stp,n_timestamps+stp,stp)
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

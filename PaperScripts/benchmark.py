# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

from convst.utils.dataset_utils import load_sktime_dataset_split
from convst.classifiers import R_DST_Ridge, R_DST_Ensemble

from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.interval_based import DrCIF
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.kernel_based import RocketClassifier

from timeit import default_timer as timer

# Define timining function
def time_pipe(pipeline, X_train, y_train):
    t0 = timer()
    pipeline.fit(X_train, y_train)
    pipeline.predict(X_train)
    t1 = timer()
    return t1-t0

# Number of validation step
n_cv = 10
# Number of parallel threads for each method
n_jobs=90

models = {'RDST Prime':R_DST_Ridge(n_jobs=n_jobs, prime_dilations=True),
          'RDST Ensemble Prime':R_DST_Ensemble(n_jobs=n_jobs, prime_dilations=True),
          'RDST':R_DST_Ridge(n_jobs=n_jobs, prime_dilations=False),
          'RDST Ensemble':R_DST_Ensemble(n_jobs=n_jobs, prime_dilations=False),
          'Rocket':RocketClassifier(n_jobs=n_jobs),
          'MultiRocket':RocketClassifier(rocket_transform='multirocket', n_jobs=n_jobs),
          'DrCIF':DrCIF(n_jobs=n_jobs), 'TDE':TemporalDictionaryEnsemble(n_jobs=n_jobs),
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
n_samples = X_train.shape[0]//3

stp = n_samples//6
lengths = np.arange(stp,(n_samples)+stp,stp)
df = pd.DataFrame(index=lengths)
for name in models.keys():
    df[name] = pd.Series(0, index=df.index)


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

stp = n_timestamps//6
lengths = np.arange(stp,n_timestamps+stp,stp)
df = pd.DataFrame(index=lengths)
for name in models.keys():
    df[name] = pd.Series(0, index=df.index)

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

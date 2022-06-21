# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:34:20 2022

@author: a694772
"""

    
from convst.utils.dataset_utils import load_sktime_dataset_split
from convst.classifiers import R_DST_Ridge, R_DST_Ensemble
import numpy as np
# In[]:
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('SemgHandMovementCh2')
print(X_train.shape)
print(X_test.shape)
# In[]:

m = R_DST_Ridge(n_shapelets=10_000).fit(X_train,y_train)
print(m.score(X_test, y_test))
print(m.transformer.values_.shape)
# In[]:
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

c = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeClassifierCV(
    )
)
"""
c = RandomForestClassifier(ccp_alpha=0.0005, n_jobs=-1)
"""
m = R_DST_Ensemble(
    n_estimators=1, n_shapelets_per_estimator=10_000, n_jobs=1,
    shapelet_sizes=[11], rot_groups=0, rot_components=20, n_samples=0.05,
    base_classifier=c
).fit(X_train,y_train)
print(m.score(X_test, y_test))


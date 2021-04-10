# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:14:16 2021

@author: Antoine
"""

# In[1]:
from CST.base_transformers.rocket import ROCKET
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
# Load GunPoint Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('GunPoint')

# Init ROCKET object
rkt = ROCKET(n_kernels=10000)

# Transforming data
X_rkt_train = rkt.fit_transform(X_train)
X_rkt_test = rkt.transform(X_test)

# Rocket Performance
rf = RandomForestClassifier(n_estimators=200, max_features=0.25, max_samples=0.5, ccp_alpha=0.01).fit(X_rkt_train, y_train)
pred = rf.predict(X_rkt_test)
print("F1-Score for ROCKET RF : {}".format(f1_score(y_test, pred, average='macro')))


# In[]:
    
from CST.shapelet_transforms import convolutional_ST
    
cst = convolutional_ST(rkt_object=rkt, ft_imps=rf.feature_importances_).fit(X_train,y_train)
X_cst_train = cst.transform(X_train)
X_cst_test = cst.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, max_features=0.25, max_samples=0.5, ccp_alpha=0.01).fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("F1-Score for CST RF : {}".format(f1_score(y_test, pred, average='macro')))


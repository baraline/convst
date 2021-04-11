# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:59:14 2021

@author: Antoine
"""


# In[1]:
from CST.base_transformers.rocket import ROCKET
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier
# Load GunPoint Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('ArrowHead')

# Init ROCKET object
rkt = ROCKET(n_kernels=10000, kernel_sizes=[9])

# Transforming data
X_rkt_train = rkt.fit_transform(X_train)
X_rkt_test = rkt.transform(X_test)

# Rocket Performance
rf = RandomForestClassifier(n_estimators=200, max_features=0.25, max_samples=0.5, ccp_alpha=0.01).fit(X_rkt_train, y_train)
pred = rf.predict(X_rkt_test)
print("F1-Score for ROCKET RF : {}".format(f1_score(y_test, pred, average='macro')))


# In[]:
    
from CST.shapelet_transforms.convolutional_ST import ConvolutionalShapeletTransformer
    
cst = ConvolutionalShapeletTransformer(rkt_object=rkt, ft_imps=rf.feature_importances_, verbose=1)
cst.fit(X_train,y_train, n_shapelet_per_combination=1, n_iter_per_comb=3, n_bins=7, percentile_select=90)
X_cst_train = cst.transform(X_train)
X_cst_test = cst.transform(X_test)

# In[]:
rf = RandomForestClassifier(n_estimators=400).fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("F1-Score for CST RF : {}".format(f1_score(y_test, pred, average='macro')))
print("Used a total of {} / {} Shapelets in model".format(rf.feature_importances_[rf.feature_importances_ > 0].size, X_cst_train.shape[1]))

# In[]:
    
#TODO : Adding some ploting to see some of the extracted shapelets

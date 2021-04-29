# -f*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:14:16 2021

@author: Antoine
"""

from CST.base_transformers.minirocket import MiniRocket
from CST.shapelet_transforms.mini_CST import MiniConvolutionalShapeletTransformer
from sklearn.linear_model import RidgeClassifierCV
from CST.utils.dataset_utils import load_sktime_dataset_split
from sklearn.metrics import f1_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load GunPoint Dataset
X_train, X_test, y_train, y_test, le = load_sktime_dataset_split('Car', normalize=True)

# Init ROCKET object
rkt = MiniRocket()

# Transforming data
X_rkt_train = rkt.fit_transform(X_train)
X_rkt_test = rkt.transform(X_test)

# Rocket Performance
rdg = RidgeClassifierCV(alphas=np.logspace(-6, 6, 20), normalize=True).fit(X_rkt_train, y_train)
pred = rdg.predict(X_rkt_test)
print("F1-Score for MINI-ROCKET: {}".format(f1_score(y_test, pred, average='macro')))
# In[]:

cst = MiniConvolutionalShapeletTransformer(verbose=1, P=[100,95,90,85], n_splits=5).fit(X_train, y_train)
X_cst_train = cst.transform(X_train)
X_cst_test = cst.transform(X_test)

rf = RandomForestClassifier(n_estimators=400, ccp_alpha=0.00).fit(X_cst_train, y_train)
pred = rf.predict(X_cst_test)
print("F1-Score for CST RF : {}".format(f1_score(y_test, pred, average='macro')))
print("Used a total of {} / {} Shapelets in model".format(rf.feature_importances_[rf.feature_importances_ > 0].size, X_cst_train.shape[1]))
# In[]:
from CST.base_transformers.shapelets import Convolutional_shapelet
import numpy as np

i_shp = np.argsort(rf.feature_importances_)[::-1][0]
#shp = Convolutional_shapelet()
grp_id = [0]
grp_id.extend(np.cumsum([len(cst.shapelets_values[k]) for k in cst.shapelets_values.keys()]))
id_grp = None
for i in range(len(grp_id)-1):
    if grp_id[i]<i_shp<grp_id[i+1]:
        id_grp = i

params = cst.shapelets_params[id_grp]
values = np.concatenate(list(cst.shapelets_values.values()),axis=0)[i_shp]
shp = Convolutional_shapelet(values=values, dilation = params[0], padding=params[2], input_ft_id=0)
# In[]:
shp.plot_loc(X_train[0,0])
